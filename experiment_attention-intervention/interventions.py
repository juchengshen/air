#!/usr/bin/env python3
"""Inference-time attention interventions for the default AIR model (Lx_H).

The shared Transformer is run with its weights unchanged, but the attention of
selected update types is replaced by one of four patterns:

  local    attention knockout that keeps only the query's neighborhood N(q)
  exclude  attention knockout that removes N(q) and keeps every other cell
  random   attention knockout that keeps a random cell set of size |N(q)|
  diffuse  every query attends uniformly to all sequence positions

Knockout masks add -1e9 to the logits of removed keys before the softmax. Every
mask keeps self-attention and full access to and from the puzzle-prefix tokens.

The answer is decoded only from z_H, so intervening on the H-update that feeds
the decoder would change the read-out directly. H interventions therefore skip
the final H-update of every inner call; L interventions apply to every L-update.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.layers import apply_rotary_pos_emb, rms_norm  # noqa: E402
from models.air.air_1net_Lx_H import AIRInnerCarry  # noqa: E402


MASK_VALUE = -1e9

# Paper condition name -> (L-update pattern, H-update pattern). None keeps natural attention.
CONDITIONS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "baseline": (None, None),
    "L_diffuse": ("diffuse", None),
    "L_excludes_neighborhood": ("exclude", None),
    "L_local_only": ("local", None),
    "L_random_only": ("random", None),
    "H_diffuse": (None, "diffuse"),
    "H_excludes_neighborhood": (None, "exclude"),
    "H_local_only": (None, "local"),
    "L_local_H_diffuse": ("local", "diffuse"),
    "L_diffuse_H_local": ("diffuse", "local"),
    "both_diffuse": ("diffuse", "diffuse"),
}

# An attention setting is either None (natural), an additive [S, S] mask, or "diffuse".
AttentionSetting = Union[None, torch.Tensor, str]


def sudoku_neighbor_fn() -> Callable[[int, int], bool]:
    """N(q) on the 9x9 board: cells sharing a row, column, or 3x3 box with q."""
    def is_neighbor(i: int, j: int) -> bool:
        ri, ci = divmod(i, 9)
        rj, cj = divmod(j, 9)
        return ri == rj or ci == cj or (ri // 3 == rj // 3 and ci // 3 == cj // 3)
    return is_neighbor


def maze_neighbor_fn(grid_size: int = 30, radius: int = 2) -> Callable[[int, int], bool]:
    """N(q) on the maze grid: the centered (2r+1)x(2r+1) window, clipped at the border."""
    def is_neighbor(i: int, j: int) -> bool:
        ri, ci = divmod(i, grid_size)
        rj, cj = divmod(j, grid_size)
        return max(abs(ri - rj), abs(ci - cj)) <= radius
    return is_neighbor


def build_knockout_masks(
    neighbor_fn: Callable[[int, int], bool],
    num_cells: int,
    *,
    puzzle_emb_len: int,
    random_seed: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build the additive [S, S] masks for the local, exclude, and random patterns,
    with S = puzzle_emb_len + num_cells.

    The random mask keeps, for each query, the same number of other cells as its
    true neighborhood, drawn once from a generator seeded with `random_seed`.
    """
    seq_len = puzzle_emb_len + num_cells
    masks = {name: torch.full((seq_len, seq_len), MASK_VALUE) for name in ("local", "exclude", "random")}
    for mask in masks.values():
        mask[:puzzle_emb_len, :] = 0.0
        mask[:, :puzzle_emb_len] = 0.0

    generator = torch.Generator().manual_seed(random_seed)
    for i in range(num_cells):
        qi = puzzle_emb_len + i
        others = [j for j in range(num_cells) if j != i]
        neighbors = [j for j in others if neighbor_fn(i, j)]
        far = [j for j in others if not neighbor_fn(i, j)]

        for mask in masks.values():
            mask[qi, qi] = 0.0
        for j in neighbors:
            masks["local"][qi, puzzle_emb_len + j] = 0.0
        for j in far:
            masks["exclude"][qi, puzzle_emb_len + j] = 0.0
        for idx in torch.randperm(len(others), generator=generator)[: len(neighbors)].tolist():
            masks["random"][qi, puzzle_emb_len + others[idx]] = 0.0

    if device is not None:
        masks = {name: mask.to(device) for name, mask in masks.items()}
    return masks


def attention_forward(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
    setting: AttentionSetting,
) -> torch.Tensor:
    """Same computation as models.layers.Attention.forward, with an optional intervention."""
    batch_size, seq_len, _ = hidden_states.shape
    qkv = attn.qkv_proj(hidden_states)
    qkv = qkv.view(batch_size, seq_len, attn.num_heads + 2 * attn.num_key_value_heads, attn.head_dim)
    query = qkv[:, :, : attn.num_heads]
    key = qkv[:, :, attn.num_heads : attn.num_heads + attn.num_key_value_heads]
    value = qkv[:, :, attn.num_heads + attn.num_key_value_heads :]

    if cos_sin is not None:
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if attn.num_heads != attn.num_key_value_heads:
        repeat = attn.num_heads // attn.num_key_value_heads
        key = key.repeat_interleave(repeat, dim=2)
        value = value.repeat_interleave(repeat, dim=2)

    q = query.permute(0, 2, 1, 3)  # [B, H, S, D]
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    if isinstance(setting, str) and setting == "diffuse":
        # Uniform attention over all positions: every query receives the mean value vector.
        out = v.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1)
    else:
        mask = setting.to(q.dtype) if isinstance(setting, torch.Tensor) else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

    out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, attn.output_size)
    return attn.o_proj(out)


def reasoning_module_forward(
    f_level: torch.nn.Module,
    hidden_states: torch.Tensor,
    input_injection: torch.Tensor,
    cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
    setting: AttentionSetting,
) -> torch.Tensor:
    """Same computation as the shared f_level (post-norm blocks), with an optional intervention."""
    x = hidden_states + input_injection
    for block in f_level.layers:
        x = rms_norm(x + attention_forward(block.self_attn, x, cos_sin, setting), variance_epsilon=block.norm_eps)
        x = rms_norm(x + block.mlp(x), variance_epsilon=block.norm_eps)
    return x


@dataclass
class InterventionSettings:
    """Attention settings for L-updates and for the non-final H-updates."""

    l_update: AttentionSetting = None
    h_update: AttentionSetting = None


def _intervened_inner_forward(
    self: torch.nn.Module,
    carry: AIRInnerCarry,
    batch: Dict[str, torch.Tensor],
    *,
    settings: InterventionSettings,
):
    """Inner forward of models.air.air_1net_Lx_H at inference, with interventions applied."""
    cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    with torch.no_grad():
        z_H, z_L = carry.z_H, carry.z_L
        for _H_step in range(self.config.H_cycles):
            for _L_step in range(self.config.L_cycles):
                if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                    z_L = reasoning_module_forward(self.f_level, z_L, z_H + input_embeddings, cos_sin, settings.l_update)
            if not (_H_step == self.config.H_cycles - 1):
                z_H = reasoning_module_forward(self.f_level, z_H, z_L, cos_sin, settings.h_update)

        z_L = reasoning_module_forward(self.f_level, z_L, z_H + input_embeddings, cos_sin, settings.l_update)
        # Final H-update feeds the decoder, so it always keeps natural attention.
        z_H = reasoning_module_forward(self.f_level, z_H, z_L, cos_sin, None)

    new_carry = AIRInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
    output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
    q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
    return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


def settings_for_condition(condition: str, knockout_masks: Dict[str, torch.Tensor]) -> InterventionSettings:
    """Translate a CONDITIONS entry into concrete attention settings."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Choose from: {', '.join(CONDITIONS)}")

    def resolve(pattern: Optional[str]) -> AttentionSetting:
        if pattern is None or pattern == "diffuse":
            return pattern
        return knockout_masks[pattern]

    l_pattern, h_pattern = CONDITIONS[condition]
    return InterventionSettings(l_update=resolve(l_pattern), h_update=resolve(h_pattern))


@contextmanager
def attention_intervention(air_model: torch.nn.Module, settings: InterventionSettings) -> Iterator[None]:
    """Temporarily route the model's inner forward through the intervened version."""
    inner = air_model.inner
    original_forward = inner.forward

    def wrapped_forward(self: torch.nn.Module, carry, batch):
        return _intervened_inner_forward(self, carry, batch, settings=settings)

    try:
        inner.forward = MethodType(wrapped_forward, inner)  # type: ignore[assignment]
        yield
    finally:
        inner.forward = original_forward  # type: ignore[assignment]
