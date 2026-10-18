#!/usr/bin/env python3
"""Evaluate one AIR checkpoint under every attention-intervention condition.

For each condition in interventions.CONDITIONS, runs the normal ACT rollout
(max halting steps) on the first N test puzzles and records exact solve and
per-cell accuracy from the final decoded z_H. On Sudoku it also counts cells in
a row/column/box violation and the number of wrong cells, from which the
aggregation script computes violations per wrong cell. Writes one JSON per
checkpoint, which aggregate_interventions.py combines across seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import interventions  # noqa: E402  -- local module
from utils.functions import load_model_class  # noqa: E402

SUPPORTED_ARCH = "air.air_1net_Lx_H@AIR"
IGNORE_LABEL_ID = 0
TASK_DEFAULTS = {
    "sudoku": {"data_path": "data/sudoku-extreme-1k-aug-1000", "eval_batch_size": 200},
    "maze": {"data_path": "data/maze-30x30-hard-1k", "eval_batch_size": 100},
}


def load_model(ckpt_path: str, dataset_metadata: Dict, eval_batch_size: int, device: torch.device) -> torch.nn.Module:
    """Build the model from the all_config.yaml next to the checkpoint and load its weights."""
    config_path = os.path.join(os.path.dirname(ckpt_path), "all_config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing all_config.yaml near checkpoint: {config_path}")
    with open(config_path, "r") as f:
        arch = dict(yaml.safe_load(f)["arch"])

    arch_name = arch.pop("name")
    arch.pop("loss", None)
    if arch_name != SUPPORTED_ARCH:
        raise ValueError(f"Interventions are implemented for {SUPPORTED_ARCH}, got {arch_name}")

    model_cfg = dict(
        **arch,
        batch_size=eval_batch_size,
        vocab_size=dataset_metadata["vocab_size"],
        seq_len=dataset_metadata["seq_len"],
        num_puzzle_identifiers=dataset_metadata["num_puzzle_identifiers"],
    )
    with torch.device(device):
        model = load_model_class(arch_name)(model_cfg)

    state_dict = torch.load(ckpt_path, map_location=device)
    # Checkpoints store the loss-head wrapper, possibly compiled: strip both prefixes.
    state_dict = {k.removeprefix("_orig_mod.").removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, assign=True)
    return model.eval()


def sudoku_violating_cells(preds: np.ndarray) -> int:
    """Count filled cells whose digit repeats in their row, column, or 3x3 box."""
    grids = preds.reshape(-1, 9, 9) - 1  # tokens: 0 pad, 1 blank, 2..10 digits -> digit 1..9
    one_hot = np.stack([grids == d for d in range(1, 10)], axis=-1)  # [N, 9, 9, 9]
    row_counts = one_hot.sum(axis=2, keepdims=True)
    col_counts = one_hot.sum(axis=1, keepdims=True)
    boxes = one_hot.reshape(-1, 3, 3, 3, 3, 9)
    box_counts = boxes.sum(axis=(2, 4), keepdims=True)
    box_counts = np.broadcast_to(box_counts, boxes.shape).reshape(one_hot.shape)
    repeated = (one_hot & ((row_counts > 1) | (col_counts > 1) | (box_counts > 1))).any(axis=-1)
    return int(repeated.sum())


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    eval_batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    """Run the ACT rollout to halting and return counts plus the final predictions."""
    num_puzzles, seq_len = inputs.shape
    exact = correct_cells = total_cells = wrong_cells = 0
    predictions: List[np.ndarray] = []

    for start in range(0, num_puzzles, eval_batch_size):
        end = min(num_puzzles, start + eval_batch_size)
        size = end - start
        # The model is built for a fixed batch size, so the last batch is zero-padded.
        batch_inputs = np.zeros((eval_batch_size, seq_len), dtype=np.int32)
        batch_labels = np.zeros((eval_batch_size, seq_len), dtype=np.int32)
        batch_inputs[:size] = inputs[start:end]
        batch_labels[:size] = labels[start:end]
        batch = {
            "inputs": torch.from_numpy(batch_inputs).to(device),
            "labels": torch.from_numpy(batch_labels).to(device),
            "puzzle_identifiers": torch.zeros(eval_batch_size, dtype=torch.int32, device=device),
        }

        with torch.device(device):
            carry = model.initial_carry(batch)
        while True:
            carry, outputs = model(carry=carry, batch=batch)
            if carry.halted.all():
                break

        pred = outputs["logits"].argmax(-1)[:size]
        target = batch["labels"][:size].long()
        valid = target != IGNORE_LABEL_ID
        correct = valid & (pred == target)
        exact += int((correct.sum(-1) == valid.sum(-1)).sum())
        correct_cells += int(correct.sum())
        total_cells += int(valid.sum())
        wrong_cells += int((valid & (pred != target)).sum())
        predictions.append(pred.cpu().numpy())

    return {
        "exact_accuracy": 100.0 * exact / num_puzzles,
        "cell_accuracy": 100.0 * correct_cells / max(1, total_cells),
        "wrong_cells": wrong_cells,
        "predictions": np.concatenate(predictions, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), required=True)
    parser.add_argument("--ckpt", required=True, help="path to a step_<N> checkpoint file")
    parser.add_argument("--data-path", default=None, help="dataset root containing test/ (default depends on --task)")
    parser.add_argument("--num-puzzles", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--conditions", default=",".join(interventions.CONDITIONS))
    parser.add_argument("--maze-radius", type=int, default=2, help="maze N(q) window radius (2 gives 5x5)")
    parser.add_argument("--random-seed", type=int, default=0, help="seed for the matched-size random mask")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()

    defaults = TASK_DEFAULTS[args.task]
    data_path = args.data_path or defaults["data_path"]
    eval_batch_size = args.eval_batch_size or defaults["eval_batch_size"]
    conditions = [c for c in args.conditions.split(",") if c.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    test_dir = os.path.join(data_path, "test")
    with open(os.path.join(test_dir, "dataset.json"), "r") as f:
        dataset_metadata = json.load(f)
    inputs = np.load(os.path.join(test_dir, "all__inputs.npy"), mmap_mode="r")[: args.num_puzzles].astype(np.int32)
    labels = np.load(os.path.join(test_dir, "all__labels.npy"), mmap_mode="r")[: args.num_puzzles].astype(np.int32)

    model = load_model(args.ckpt, dataset_metadata, eval_batch_size, device)
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Evaluating {inputs.shape[0]} {args.task} test puzzles on {device}")

    if args.task == "sudoku":
        neighbor_fn = interventions.sudoku_neighbor_fn()
    else:
        neighbor_fn = interventions.maze_neighbor_fn(grid_size=int(round(dataset_metadata["seq_len"] ** 0.5)), radius=args.maze_radius)
    knockout_masks = interventions.build_knockout_masks(
        neighbor_fn,
        dataset_metadata["seq_len"],
        puzzle_emb_len=model.inner.puzzle_emb_len,
        random_seed=args.random_seed,
        device=device,
    )

    results: Dict[str, Dict[str, float]] = {}
    for condition in conditions:
        started = time.time()
        settings = interventions.settings_for_condition(condition, knockout_masks)
        with interventions.attention_intervention(model, settings):
            metrics = evaluate(model, inputs, labels, eval_batch_size, device)

        predictions = metrics.pop("predictions")
        if args.task == "sudoku":
            metrics["violating_cells"] = sudoku_violating_cells(predictions)
        results[condition] = metrics

        summary = f"  {condition:24s} exact={metrics['exact_accuracy']:6.2f}  cell={metrics['cell_accuracy']:6.2f}"
        if "violating_cells" in metrics:
            summary += f"  violations/wrong={metrics['violating_cells'] / max(1, metrics['wrong_cells']):.3f}"
        print(f"{summary}  ({time.time() - started:.0f}s)")

    payload = {
        "task": args.task,
        "checkpoint_path": args.ckpt,
        "num_puzzles": int(inputs.shape[0]),
        "maze_radius": args.maze_radius if args.task == "maze" else None,
        "random_seed": args.random_seed,
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
