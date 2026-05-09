"""Shared setup and tracing for Maze visual freeze/decode experiments."""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Iterable
import yaml

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AIR_ROOT = os.path.dirname(SCRIPT_DIR)
if AIR_ROOT not in sys.path:
    sys.path.insert(0, AIR_ROOT)
os.chdir(AIR_ROOT)
os.environ["DISABLE_COMPILE"] = "1"

from pretrain import PretrainConfig, init_train_state, create_dataloader
from models.air.air_1net_Lx_H import AIR_Inner, AIRInnerCarry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE = 30
DEFAULT_NUM_QUESTIONS = 1000

# Token mapping (from build_maze_dataset.py CHARSET "# SGo"):
WALL_TOKEN = 1      # '#'
PATH_TOKEN = 2      # ' ' (open path)
START_TOKEN = 3     # 'S'
GOAL_TOKEN = 4      # 'G'
SOLUTION_TOKEN = 5  # 'o' (solution path)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    AIR_ROOT, "checkpoints", "Maze", "air_Lx_H_bs768_lr1e4_20k_epochs"
)
DEFAULT_CHECKPOINT_FILE = os.path.join(DEFAULT_CHECKPOINT_DIR, "step_26040")


def apply_maze_l_update(
    inner: AIR_Inner,
    z_L: torch.Tensor,
    z_H: torch.Tensor,
    input_embeddings: torch.Tensor,
    seq_info: dict,
) -> torch.Tensor:
    """Apply architecture-specific L update for Maze variants."""
    module_name = inner.__class__.__module__.rsplit(".", 1)[-1]

    if module_name == "air_1net_L2x_Hx":
        return inner.f_level(z_L + input_embeddings, z_H + input_embeddings, **seq_info)

    return inner.f_level(z_L, z_H + input_embeddings, **seq_info)


def apply_maze_h_update(
    inner: AIR_Inner,
    z_H: torch.Tensor,
    z_L: torch.Tensor,
    input_embeddings: torch.Tensor,
    seq_info: dict,
) -> torch.Tensor:
    """Apply architecture-specific H update for Maze variants."""
    module_name = inner.__class__.__module__.rsplit(".", 1)[-1]

    if module_name == "air_1net_L2x_Hx":
        return inner.f_level(z_H + input_embeddings, z_L, **seq_info)

    return inner.f_level(z_H, z_L, **seq_info)


def resolve_checkpoint_paths() -> tuple[str, str]:
    """Resolve checkpoint and config paths, allowing override via environment."""
    checkpoint_file = os.environ.get("AIR_MAZE_CKPT_PATH", DEFAULT_CHECKPOINT_FILE)
    if not os.path.isabs(checkpoint_file):
        checkpoint_file = os.path.join(AIR_ROOT, checkpoint_file)
    checkpoint_file = os.path.abspath(checkpoint_file)

    checkpoint_dir = os.path.dirname(checkpoint_file)
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found next to checkpoint: {config_path}")

    return checkpoint_file, config_path


def run_inner_with_tracing(
    inner: AIR_Inner,
    carry,
    batch: dict,
    *,
    freeze_z_H: bool = False,
    freeze_z_L: bool = False,
):
    """Run inner loop step-by-step. Optionally freeze z_H or z_L."""
    seq_info = {"cos_sin": inner.rotary_emb() if hasattr(inner, "rotary_emb") else None}
    input_embeddings = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    plen = inner.puzzle_emb_len
    H_cycles = inner.config.H_cycles
    L_cycles = inner.config.L_cycles

    z_H = carry.z_H.clone()
    z_L = carry.z_L.clone()
    traces = []

    def decode(z):
        logits = inner.lm_head(z)[:, plen:].float().cpu()
        pred = logits.argmax(dim=-1).numpy()
        return pred

    with torch.no_grad():
        for H_step in range(H_cycles):
            for L_step in range(L_cycles):
                if not ((H_step == H_cycles - 1) and (L_step == L_cycles - 1)):
                    if not freeze_z_L:
                        z_L = apply_maze_l_update(inner, z_L, z_H, input_embeddings, seq_info)
                    traces.append({
                        "stage": "L", "H_step": H_step, "L_step": L_step,
                        "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
                        "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
                    })
            if not (H_step == H_cycles - 1):
                if not freeze_z_H:
                    z_H = apply_maze_h_update(inner, z_H, z_L, input_embeddings, seq_info)
                traces.append({
                    "stage": "H", "H_step": H_step, "L_step": None,
                    "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
                    "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
                })

    if not freeze_z_L:
        z_L = apply_maze_l_update(inner, z_L, z_H, input_embeddings, seq_info)
    traces.append({
        "stage": "L", "H_step": H_cycles - 1, "L_step": L_cycles - 1,
        "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
        "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
    })
    if not freeze_z_H:
        z_H = apply_maze_h_update(inner, z_H, z_L, input_embeddings, seq_info)
    traces.append({
        "stage": "H", "H_step": H_cycles - 1, "L_step": None,
        "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
        "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
    })

    new_carry = AIRInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
    logits = inner.lm_head(z_H)[:, plen:]
    q_logits = inner.q_head(z_H[:, 0]).to(torch.float32)
    q_halt, q_continue = q_logits[..., 0], q_logits[..., 1]
    return traces, new_carry, logits, q_halt, q_continue


def _load_config_model_and_eval_loader():
    """Load Maze config, model, and eval loader."""
    checkpoint_file, config_path = resolve_checkpoint_paths()
    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    train_loader, train_metadata = create_dataloader(
        config, "train", 0, 1, test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size
    )
    eval_loader, _ = create_dataloader(
        config, "test", 0, 1, test_set_mode=True, epochs_per_iter=1, global_batch_size=1
    )

    train_state = init_train_state(config, train_metadata, world_size=1)
    if DEVICE.type == "cpu":
        train_state.model.to(DEVICE)
    ckpt = torch.load(checkpoint_file, map_location=DEVICE)
    try:
        train_state.model.load_state_dict(ckpt, assign=True)
    except Exception:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}, assign=True
        )
    train_state.model.eval()
    air = train_state.model.model
    inner = air.inner
    return config, air, inner, eval_loader


def run_act_with_tracing(
    air,
    inner,
    batch: dict,
    *,
    freeze_z_H: bool = False,
    freeze_z_L: bool = False,
):
    """Run the full ACT loop with tracing for one Maze batch."""
    with torch.device(DEVICE.type):
        carry = air.initial_carry(batch)
    act_step_traces = []
    halt_max_steps = air.config.halt_max_steps

    with torch.inference_mode():
        for act_step in range(halt_max_steps):
            new_inner_carry = air.inner.reset_carry(carry.halted, carry.inner_carry)
            new_steps = torch.where(carry.halted, 0, carry.steps)
            new_current_data = {
                k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
                for k, v in carry.current_data.items()
            }
            traces, new_inner_carry, logits, _, _ = run_inner_with_tracing(
                inner,
                new_inner_carry,
                new_current_data,
                freeze_z_H=freeze_z_H,
                freeze_z_L=freeze_z_L,
            )
            act_step_traces.append({"act_step": act_step, "traces": traces, "logits": logits.detach()})
            new_steps = new_steps + 1
            halted = new_steps >= halt_max_steps
            carry = type(carry)(
                inner_carry=new_inner_carry, steps=new_steps, halted=halted, current_data=new_current_data
            )
            if halted.all():
                break

    return act_step_traces


def _load_and_run(batch_index: int = 0, freeze_z_H: bool = False, freeze_z_L: bool = False):
    """Shared implementation for load_config_model_batch variants."""
    config, air, inner, eval_loader = _load_config_model_and_eval_loader()

    eval_iter = iter(eval_loader)
    for _ in range(batch_index + 1):
        _set_name, batch, _ = next(eval_iter)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    act_step_traces = run_act_with_tracing(
        air, inner, batch, freeze_z_H=freeze_z_H, freeze_z_L=freeze_z_L
    )
    return config, air, inner, batch, act_step_traces


def load_config_model_batch(batch_index: int = 0):
    """Load config, model, one batch; run full ACT loop with tracing."""
    return _load_and_run(batch_index=batch_index, freeze_z_H=False, freeze_z_L=False)


def load_config_model_batch_with_options(
    batch_index: int = 0,
    freeze_z_H: bool = False,
    freeze_z_L: bool = False,
):
    """Same as load_config_model_batch but with freeze options."""
    return _load_and_run(batch_index=batch_index, freeze_z_H=freeze_z_H, freeze_z_L=freeze_z_L)


def load_config_model_batches(
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    *,
    freeze_z_H: bool = False,
    freeze_z_L: bool = False,
):
    """Load Maze config/model once, then trace the first num_questions eval puzzles."""
    config, air, inner, eval_loader = _load_config_model_and_eval_loader()
    eval_iter = iter(eval_loader)
    question_runs = []
    for batch_index in range(num_questions):
        try:
            _set_name, batch, _ = next(eval_iter)
        except StopIteration:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        act_step_traces = run_act_with_tracing(
            air, inner, batch, freeze_z_H=freeze_z_H, freeze_z_L=freeze_z_L
        )
        question_runs.append(
            {
                "batch_index": batch_index,
                "batch": batch,
                "act_step_traces": act_step_traces,
            }
        )
    return config, air, inner, question_runs


def flatten_recursive_traces(act_step_traces: list[dict]) -> list[dict]:
    """Flatten nested ACT traces into one recursive-step list."""
    steps = []
    for act_data in act_step_traces:
        act_step = act_data["act_step"]
        for sub_step, trace in enumerate(act_data["traces"]):
            steps.append(
                {
                    "act_step": act_step,
                    "sub_step": sub_step,
                    "stage": trace["stage"],
                    "H_step": trace["H_step"],
                    "L_step": trace["L_step"],
                    "pred_H": trace["pred_H"][0],
                    "pred_L": trace["pred_L"][0],
                }
            )
    return steps


def batch_grid(batch: dict, key: str) -> np.ndarray:
    """Return a (GRID_SIZE, GRID_SIZE) numpy grid from a batch tensor field."""
    return batch[key].detach().cpu().numpy().reshape(-1, GRID_SIZE, GRID_SIZE)[0]


def align_series(series_list: Iterable[Iterable[float]], fill_value: float = np.nan) -> np.ndarray:
    """Pad variable-length 1D series into a dense array."""
    series_list = [np.asarray(series, dtype=np.float64) for series in series_list]
    if not series_list:
        return np.zeros((0, 0), dtype=np.float64)
    max_len = max(len(series) for series in series_list)
    aligned = np.full((len(series_list), max_len), fill_value, dtype=np.float64)
    for idx, series in enumerate(series_list):
        aligned[idx, : len(series)] = series
    return aligned


def aggregate_series(series_list: Iterable[Iterable[float]]) -> dict:
    """Return count, mean, std, and sem for aligned 1D series."""
    aligned = align_series(series_list)
    if aligned.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {"aligned": aligned, "count": empty, "mean": empty, "std": empty, "sem": empty}

    mask = ~np.isnan(aligned)
    count = mask.sum(axis=0).astype(np.int32)
    mean = np.divide(
        np.nansum(aligned, axis=0),
        np.maximum(count, 1),
        out=np.zeros(aligned.shape[1], dtype=np.float64),
        where=count > 0,
    )
    std = np.zeros(aligned.shape[1], dtype=np.float64)
    sem = np.zeros(aligned.shape[1], dtype=np.float64)
    for idx in range(aligned.shape[1]):
        values = aligned[mask[:, idx], idx]
        if values.size == 0:
            continue
        if values.size == 1:
            std[idx] = 0.0
            sem[idx] = 0.0
            continue
        std[idx] = float(np.std(values, ddof=1))
        sem[idx] = std[idx] / math.sqrt(values.size)
    return {"aligned": aligned, "count": count, "mean": mean, "std": std, "sem": sem}


def summarize_scalars(values: Iterable[float]) -> dict:
    """Return scalar mean/std/sem summary."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0}
    if arr.size == 1:
        return {"n": 1, "mean": float(arr[0]), "std": 0.0, "sem": 0.0}
    std = float(np.std(arr, ddof=1))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": std,
        "sem": float(std / math.sqrt(arr.size)),
    }



def metrics_table_rows(metric_to_values: dict[str, Iterable[float]]) -> list[list[object]]:
    """Build Markdown-friendly rows from scalar metric collections."""
    rows = []
    for metric_name, values in metric_to_values.items():
        summary = summarize_scalars(values)
        rows.append(
            [
                metric_name,
                summary["n"],
                f"{summary['mean']:.4f}",
                f"{summary['std']:.4f}",
                f"{summary['sem']:.4f}",
            ]
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a Markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_markdown_table(path: str, headers: list[str], rows: list[list[object]], title: str | None = None):
    """Write a Markdown table to disk."""
    parts = []
    if title:
        parts.append(f"# {title}")
        parts.append("")
    parts.append(markdown_table(headers, rows))
    parts.append("")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def sanitize_for_json(obj):
    """Convert numpy and tensor values to JSON-serializable Python types."""
    if isinstance(obj, torch.Tensor):
        return sanitize_for_json(obj.detach().cpu().numpy())
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    return obj


def write_json(path: str, payload: dict):
    """Write a JSON payload with numpy-safe serialization."""
    with open(path, "w") as f:
        json.dump(sanitize_for_json(payload), f, indent=2)
