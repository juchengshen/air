#!/usr/bin/env python3
"""Sudoku attention-extraction helpers used by generate_bar_data.py:
checkpoint/config/model/test-loader bootstrap, plus L/H attention map
capture and per-stage decoded-grid extraction for selected ACT steps."""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.layers import apply_rotary_pos_emb
from pretrain import PretrainConfig, create_dataloader, init_train_state


BLANK_TOKEN = 1


@dataclass
class AttentionContext:
    """Loaded runtime context for attention extraction."""

    checkpoint_path: str
    config: PretrainConfig
    model: torch.nn.Module
    eval_loader: Iterable[Tuple[str, Dict[str, torch.Tensor], int]]
    device: torch.device


def resolve_checkpoint_path(path: str) -> str:
    """
    Resolve a checkpoint path.

    If `path` is a directory, picks the latest `step_<N>` file.
    """
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        pattern = re.compile(r"^step_(\d+)$")
        candidates: List[Tuple[int, str]] = []
        for name in os.listdir(path):
            match = pattern.match(name)
            if match is None:
                continue
            full = os.path.join(path, name)
            if os.path.isfile(full):
                candidates.append((int(match.group(1)), full))
        if not candidates:
            raise FileNotFoundError(f"No step_<N> checkpoint files in: {path}")
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint_and_test_loader(
    checkpoint: str,
    *,
    device: Optional[torch.device] = None,
    eval_batch_size: int = 1,
) -> AttentionContext:
    """
    Load config/checkpoint/model and create test loader from existing training code.
    """
    ckpt_path = resolve_checkpoint_path(checkpoint)
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing all_config.yaml near checkpoint: {config_path}")

    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        rank=0,
        world_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=eval_batch_size,
    )

    train_state = init_train_state(config, eval_metadata, world_size=1)
    ckpt_payload = torch.load(ckpt_path, map_location=runtime_device)
    try:
        train_state.model.load_state_dict(ckpt_payload, assign=True)
    except Exception:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in ckpt_payload.items()},
            assign=True,
        )

    train_state.model.eval()
    if runtime_device.type == "cpu":
        train_state.model.to(runtime_device)

    return AttentionContext(
        checkpoint_path=ckpt_path,
        config=config,
        model=train_state.model.model,
        eval_loader=eval_loader,
        device=runtime_device,
    )


def build_inner_call_schedule(h_cycles: int, l_cycles: int) -> List[Tuple[str, int, Optional[int]]]:
    """Build the inner-call stage schedule used by capture_lh_maps_all_steps."""
    schedule: List[Tuple[str, int, Optional[int]]] = []
    for h_step in range(h_cycles):
        for l_step in range(l_cycles):
            if not ((h_step == h_cycles - 1) and (l_step == l_cycles - 1)):
                schedule.append(("L", h_step, l_step))
        if h_step != h_cycles - 1:
            schedule.append(("H", h_step, None))
    schedule.append(("L", h_cycles - 1, l_cycles - 1))
    schedule.append(("H", h_cycles - 1, None))
    return schedule


def neighborhood_indices(query_idx: int, *, include_self: bool = True) -> np.ndarray:
    """
    Sudoku neighborhood indices (row ∪ col ∪ box) for query cell index [0..80].
    """
    if query_idx < 0 or query_idx >= 81:
        raise ValueError(f"query_idx out of range: {query_idx}")
    r = query_idx // 9
    c = query_idx % 9
    row = {r * 9 + j for j in range(9)}
    col = {i * 9 + c for i in range(9)}
    br = (r // 3) * 3
    bc = (c // 3) * 3
    box = {(br + i) * 9 + (bc + j) for i in range(3) for j in range(3)}
    union = sorted(row | col | box)
    if include_self:
        return np.asarray(union, dtype=np.int64)
    return np.asarray([idx for idx in union if idx != query_idx], dtype=np.int64)


def violation_neighbor_count(query_idx: int, violation_mask: np.ndarray) -> int:
    """Count violated Sudoku-neighborhood cells around q (excluding q)."""
    neigh = neighborhood_indices(query_idx, include_self=False)
    return int(violation_mask.reshape(-1)[neigh].sum())


def compute_maps_for_heads(
    *,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
    head_indices: Sequence[int],
    puzzle_emb_len: int,
) -> Dict[int, np.ndarray]:
    """
    Compute attention maps for selected heads at one layer call.

    
    """
    attn = layer.self_attn  # type: ignore[attr-defined]

    batch_size, seq_len, _ = hidden_states.shape
    qkv = attn.qkv_proj(hidden_states)
    qkv = qkv.view(
        batch_size,
        seq_len,
        attn.num_heads + 2 * attn.num_key_value_heads,
        attn.head_dim,
    )
    query = qkv[:, :, : attn.num_heads]
    key = qkv[:, :, attn.num_heads : attn.num_heads + attn.num_key_value_heads]

    if cos_sin is not None:
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if attn.num_heads != attn.num_key_value_heads:
        if attn.num_heads % attn.num_key_value_heads != 0:
            raise ValueError("num_heads must be multiple of num_key_value_heads")
        repeat = attn.num_heads // attn.num_key_value_heads
        key = key.repeat_interleave(repeat, dim=2)

    for h in head_indices:
        if h < 0 or h >= attn.num_heads:
            raise ValueError(f"head {h} out of range [0, {attn.num_heads - 1}]")

    q = query.to(torch.float32)
    k = key.to(torch.float32)
    scores = torch.einsum("bshd,bthd->bhst", q, k) / math.sqrt(attn.head_dim)
    probs = torch.softmax(scores, dim=-1)

    out: Dict[int, np.ndarray] = {}
    for h in head_indices:
        mat = probs[0, h, puzzle_emb_len:, puzzle_emb_len:].detach().cpu().numpy()
        out[int(h)] = mat
    return out


def capture_lh_maps_all_steps(
    *,
    air_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    max_act_steps: int,
    h_step: int,
    l_step: int,
    layer_idx: int,
    head_indices: Sequence[int],
    selected_act_steps: Optional[Set[int]] = None,
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    """
    Capture stage-targeted L/H attention maps for selected ACT steps.

    Returns:
      out[act_step][head]["L"|"H"] = [81,81] attention map
    """
    inner = air_model.inner
    if layer_idx < 0 or layer_idx >= len(inner.f_level.layers):
        raise ValueError(f"layer_idx {layer_idx} out of range [0, {len(inner.f_level.layers)-1}]")
    if h_step < 0 or h_step >= inner.config.H_cycles:
        raise ValueError(f"h_step {h_step} out of range [0, {inner.config.H_cycles-1}]")
    if l_step < 0 or l_step >= inner.config.L_cycles:
        raise ValueError(f"l_step {l_step} out of range [0, {inner.config.L_cycles-1}]")

    schedule = build_inner_call_schedule(inner.config.H_cycles, inner.config.L_cycles)
    target_l = ("L", h_step, l_step)
    target_h = ("H", h_step, None)
    if target_l not in schedule:
        raise ValueError(f"Invalid target L stage: {target_l}")
    if target_h not in schedule:
        raise ValueError(f"Invalid target H stage: {target_h}")

    batch_device = batch["inputs"].device
    with torch.device(batch_device.type):
        carry = air_model.initial_carry(batch)

    result: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {}

    with torch.inference_mode():
        for act_step in range(max_act_steps):
            should_store = selected_act_steps is None or act_step in selected_act_steps
            if should_store:
                result.setdefault(act_step, {h: {} for h in head_indices})

            call_idx = 0
            original_forward = inner.f_level.forward

            new_inner_carry = inner.reset_carry(carry.halted, carry.inner_carry)
            new_steps = torch.where(carry.halted, 0, carry.steps)
            new_current_data = {
                k: torch.where(
                    carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                    batch[k],
                    v,
                )
                for k, v in carry.current_data.items()
            }

            def wrapped_forward(
                self: torch.nn.Module,
                hidden_states: torch.Tensor,
                input_injection: torch.Tensor,
                **kwargs: Any,
            ) -> torch.Tensor:
                nonlocal call_idx
                if call_idx >= len(schedule):
                    raise RuntimeError(
                        f"Inner call index {call_idx} exceeded schedule length {len(schedule)}."
                    )
                stage_info = schedule[call_idx]
                call_idx += 1

                x = hidden_states + input_injection
                for li, layer in enumerate(self.layers):  # type: ignore[attr-defined]
                    if should_store and li == layer_idx and (stage_info == target_l or stage_info == target_h):
                        maps = compute_maps_for_heads(
                            layer=layer,
                            hidden_states=x,
                            cos_sin=kwargs.get("cos_sin"),
                            head_indices=head_indices,
                            puzzle_emb_len=inner.puzzle_emb_len,
                        )
                        stage_name = "L" if stage_info == target_l else "H"
                        for h, m in maps.items():
                            result[act_step][h][stage_name] = m
                    x = layer(hidden_states=x, **kwargs)
                return x

            try:
                inner.f_level.forward = MethodType(wrapped_forward, inner.f_level)  # type: ignore[assignment]
                new_inner_carry, _logits, _q = inner(new_inner_carry, new_current_data)
            finally:
                inner.f_level.forward = original_forward  # type: ignore[assignment]

            new_steps = new_steps + 1
            halted = new_steps >= air_model.config.halt_max_steps
            carry = type(carry)(
                inner_carry=new_inner_carry,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            )

            if should_store:
                missing = [
                    (h, stage_name)
                    for h in head_indices
                    for stage_name in ("L", "H")
                    if stage_name not in result[act_step][h]
                ]
                if missing:
                    raise RuntimeError(f"Missing captures at act_step={act_step}: {missing}")

    return result


def decode_stage_grids_all_steps(
    *,
    air_model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    max_act_steps: int,
    h_step: int,
    l_step: int,
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """
    Decode stage-targeted z_L/z_H grids across ACT steps.

    Returns:
      out[act_step]["L"]["pred_L"|"pred_H"]
      out[act_step]["H"]["pred_L"|"pred_H"]
    """
    inner = air_model.inner
    h_cycles = int(inner.config.H_cycles)
    l_cycles = int(inner.config.L_cycles)
    if not (0 <= h_step < h_cycles):
        raise ValueError(f"h_step {h_step} out of range [0, {h_cycles - 1}]")
    if not (0 <= l_step < l_cycles):
        raise ValueError(f"l_step {l_step} out of range [0, {l_cycles - 1}]")

    plen = int(inner.puzzle_emb_len)

    def decode_grid(z: torch.Tensor) -> np.ndarray:
        logits = inner.lm_head(z)[:, plen:]
        pred = torch.argmax(logits, dim=-1)[0].detach().cpu().numpy().reshape(9, 9)
        return pred.astype(np.int64)

    batch_device = batch["inputs"].device
    with torch.device(batch_device.type):
        carry = air_model.initial_carry(batch)

    out: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    with torch.inference_mode():
        for act_step in range(max_act_steps):
            new_inner_carry = inner.reset_carry(carry.halted, carry.inner_carry)
            new_steps = torch.where(carry.halted, 0, carry.steps)
            new_current_data = {
                k: torch.where(
                    carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                    batch[k],
                    v,
                )
                for k, v in carry.current_data.items()
            }

            seq_info = {"cos_sin": inner.rotary_emb() if hasattr(inner, "rotary_emb") else None}
            input_embeddings = inner._input_embeddings(
                new_current_data["inputs"], new_current_data["puzzle_identifiers"]
            )

            z_h = new_inner_carry.z_H
            z_l = new_inner_carry.z_L
            stage_store: Dict[str, Dict[str, np.ndarray]] = {}

            for hs in range(h_cycles):
                for ls in range(l_cycles):
                    if not ((hs == h_cycles - 1) and (ls == l_cycles - 1)):
                        z_l = inner.f_level(z_l, z_h + input_embeddings, **seq_info)
                        if hs == h_step and ls == l_step:
                            stage_store["L"] = {"pred_L": decode_grid(z_l), "pred_H": decode_grid(z_h)}
                if hs != h_cycles - 1:
                    z_h = inner.f_level(z_h, z_l, **seq_info)
                    if hs == h_step:
                        stage_store["H"] = {"pred_L": decode_grid(z_l), "pred_H": decode_grid(z_h)}

            z_l_new = inner.f_level(z_l, z_h + input_embeddings, **seq_info)
            if h_step == h_cycles - 1 and l_step == l_cycles - 1:
                stage_store["L"] = {"pred_L": decode_grid(z_l_new), "pred_H": decode_grid(z_h)}
            z_h_new = inner.f_level(z_h, z_l_new, **seq_info)
            if h_step == h_cycles - 1:
                stage_store["H"] = {"pred_L": decode_grid(z_l_new), "pred_H": decode_grid(z_h_new)}

            if "L" not in stage_store or "H" not in stage_store:
                raise RuntimeError(f"Missing decoded stages at act_step={act_step}.")
            out[act_step] = stage_store

            updated_carry = type(new_inner_carry)(z_H=z_h_new.detach(), z_L=z_l_new.detach())
            new_steps = new_steps + 1
            halted = new_steps >= air_model.config.halt_max_steps
            carry = type(carry)(
                inner_carry=updated_carry,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            )

    return out


