#!/usr/bin/env python3
"""
Freeze z_L ablation.

If we stop updating z_L and let only z_H recurse, does the mask still shift?

This tests whether mask motion can still be driven by H dynamics when L is held fixed.
Blank token id is 1 (model/dataset vocab); mask_blank uses BLANK_TOKEN from commons.

Run from AIR_code root: python experiment_visual-sudoku-decoded-freeze/sudoku_freeze_zL.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from commons import (
    BLANK_TOKEN,
    DEVICE,
    DEFAULT_NUM_QUESTIONS,
    _load_config_model_and_eval_loader,
    batch_grid,
    flatten_recursive_traces,
    markdown_table,
    run_act_with_tracing,
    write_json,
    write_markdown_table,
)

DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "test_sudoku_new_L2x_Hx_20k_epochs")
OUT_DIR = os.environ.get("AIR_SUDOKU_TEST_OUT_DIR", DEFAULT_OUT_DIR)
if not os.path.isabs(OUT_DIR):
    OUT_DIR = os.path.join(SCRIPT_DIR, OUT_DIR)
OUT_DIR = os.path.abspath(OUT_DIR)
NUM_QUESTIONS = int(os.environ.get("AIR_SUDOKU_NUM_QUESTIONS", str(DEFAULT_NUM_QUESTIONS)))
os.makedirs(OUT_DIR, exist_ok=True)


def mask_blank(grid: np.ndarray) -> np.ndarray:
    """True where cell is blank (token id 1). grid shape (9,9)."""
    return (grid == BLANK_TOKEN).astype(np.uint8)


def transition_metrics(prev_grid: np.ndarray, curr_grid: np.ndarray, mask_fn) -> dict[str, int]:
    prev_mask = mask_fn(prev_grid).astype(bool)
    curr_mask = mask_fn(curr_grid).astype(bool)
    mask_flip = np.logical_xor(prev_mask, curr_mask)
    token_change = prev_grid != curr_grid
    content_change = np.logical_and(token_change, np.logical_not(mask_flip))
    return {
        "mask_flips": int(mask_flip.sum()),
        "content_changes": int(content_change.sum()),
        "token_changes": int(token_change.sum()),
    }


def flips_per_step_from_steps(step_grids: list[np.ndarray]) -> list[int]:
    return [
        transition_metrics(prev_grid, curr_grid, mask_blank)["mask_flips"]
        for prev_grid, curr_grid in zip(step_grids, step_grids[1:])
    ]


def content_changes_per_step_from_steps(step_grids: list[np.ndarray], mask_fn) -> list[int]:
    return [
        transition_metrics(prev_grid, curr_grid, mask_fn)["content_changes"]
        for prev_grid, curr_grid in zip(step_grids, step_grids[1:])
    ]


def exact_match_accuracy(pred_grid: np.ndarray, label_grid: np.ndarray) -> float:
    """Exact sequence accuracy over supervised cells (1.0 if all are correct, else 0.0)."""
    pred_grid = np.asarray(pred_grid, dtype=np.int32)
    label_grid = np.asarray(label_grid, dtype=np.int32)
    supervised_mask = label_grid >= 0
    if not supervised_mask.any():
        return 0.0
    return float(np.all(pred_grid[supervised_mask] == label_grid[supervised_mask]))


class RunningScalarStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float):
        x = float(value)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0}
        if self.n == 1:
            return {"n": 1, "mean": self.mean, "std": 0.0, "sem": 0.0}
        std = float(np.sqrt(self.m2 / (self.n - 1)))
        return {
            "n": int(self.n),
            "mean": float(self.mean),
            "std": std,
            "sem": float(std / np.sqrt(self.n)),
        }


class RunningSeriesStats:
    def __init__(self):
        self.count = np.zeros(0, dtype=np.int64)
        self.sum = np.zeros(0, dtype=np.float64)
        self.sumsq = np.zeros(0, dtype=np.float64)

    def update(self, series: list[int] | list[float]):
        arr = np.asarray(series, dtype=np.float64)
        n = arr.size
        if n == 0:
            return
        if n > self.count.size:
            grow = n - self.count.size
            self.count = np.pad(self.count, (0, grow), constant_values=0)
            self.sum = np.pad(self.sum, (0, grow), constant_values=0.0)
            self.sumsq = np.pad(self.sumsq, (0, grow), constant_values=0.0)
        self.count[:n] += 1
        self.sum[:n] += arr
        self.sumsq[:n] += arr * arr

    def summary(self) -> dict:
        if self.count.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return {"count": empty.astype(np.int32), "mean": empty, "std": empty, "sem": empty}

        mean = np.divide(
            self.sum,
            np.maximum(self.count, 1),
            out=np.zeros_like(self.sum),
            where=self.count > 0,
        )
        std = np.zeros_like(self.sum)
        valid = self.count > 1
        numer = self.sumsq[valid] - (self.sum[valid] ** 2) / self.count[valid]
        var = np.maximum(numer / (self.count[valid] - 1), 0.0)
        std[valid] = np.sqrt(var)
        sem = np.zeros_like(self.sum)
        nonzero = self.count > 0
        sem[nonzero] = std[nonzero] / np.sqrt(self.count[nonzero])

        return {
            "count": self.count.astype(np.int32),
            "mean": mean,
            "std": std,
            "sem": sem,
        }


def _table_rows_from_stats(metric_order: list[str], scalar_stats: dict[str, RunningScalarStats]) -> list[list[object]]:
    rows = []
    for metric_name in metric_order:
        s = scalar_stats[metric_name].summary()
        rows.append(
            [
                metric_name,
                s["n"],
                f"{s['mean']:.4f}",
                f"{s['std']:.4f}",
                f"{s['sem']:.4f}",
            ]
        )
    return rows


def main():
    print(f"Running normal + freeze z_L over first {NUM_QUESTIONS} eval questions (streaming).")
    config, air, inner, eval_loader = _load_config_model_and_eval_loader()
    eval_iter = iter(eval_loader)

    normal_flip_series = RunningSeriesStats()
    freeze_flip_series = RunningSeriesStats()
    normal_content_series = RunningSeriesStats()
    freeze_content_series = RunningSeriesStats()

    metric_order = [
        "Total mask flips: normal",
        "Total mask flips: freeze z_L",
        "Delta total flips (freeze - normal)",
        "Total number changes: normal",
        "Total number changes: freeze z_L",
        "Delta total number changes (freeze - normal)",
        "Mean flips per step: normal",
        "Mean flips per step: freeze z_L",
        "Mean number changes per step: normal",
        "Mean number changes per step: freeze z_L",
        "Exact accuracy: normal",
        "Exact accuracy: freeze z_L",
    ]
    scalar_stats = {name: RunningScalarStats() for name in metric_order}

    processed = 0
    for _batch_index in range(NUM_QUESTIONS):
        try:
            _set_name, batch, _ = next(eval_iter)
        except StopIteration:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch_grid(batch, "labels")

        normal_traces = run_act_with_tracing(air, inner, batch)
        freeze_traces = run_act_with_tracing(air, inner, batch, freeze_z_L=True)

        normal_steps = [step["pred_H"] for step in flatten_recursive_traces(normal_traces)]
        freeze_steps = [step["pred_H"] for step in flatten_recursive_traces(freeze_traces)]

        per_step_normal = flips_per_step_from_steps(normal_steps)
        per_step_freeze = flips_per_step_from_steps(freeze_steps)
        content_normal = content_changes_per_step_from_steps(normal_steps, mask_blank)
        content_freeze = content_changes_per_step_from_steps(freeze_steps, mask_blank)
        total_normal_flips = float(sum(per_step_normal))
        total_freeze_flips = float(sum(per_step_freeze))
        total_normal_changes = float(sum(content_normal))
        total_freeze_changes = float(sum(content_freeze))
        mean_normal_flips = float(np.mean(per_step_normal)) if per_step_normal else 0.0
        mean_freeze_flips = float(np.mean(per_step_freeze)) if per_step_freeze else 0.0
        mean_normal_changes = float(np.mean(content_normal)) if content_normal else 0.0
        mean_freeze_changes = float(np.mean(content_freeze)) if content_freeze else 0.0
        final_normal = normal_steps[-1] if normal_steps else labels
        final_freeze = freeze_steps[-1] if freeze_steps else labels
        exact_normal = exact_match_accuracy(final_normal, labels)
        exact_freeze = exact_match_accuracy(final_freeze, labels)

        normal_flip_series.update(per_step_normal)
        freeze_flip_series.update(per_step_freeze)
        normal_content_series.update(content_normal)
        freeze_content_series.update(content_freeze)

        scalar_stats["Total mask flips: normal"].update(total_normal_flips)
        scalar_stats["Total mask flips: freeze z_L"].update(total_freeze_flips)
        scalar_stats["Delta total flips (freeze - normal)"].update(total_freeze_flips - total_normal_flips)
        scalar_stats["Total number changes: normal"].update(total_normal_changes)
        scalar_stats["Total number changes: freeze z_L"].update(total_freeze_changes)
        scalar_stats["Delta total number changes (freeze - normal)"].update(total_freeze_changes - total_normal_changes)
        scalar_stats["Mean flips per step: normal"].update(mean_normal_flips)
        scalar_stats["Mean flips per step: freeze z_L"].update(mean_freeze_flips)
        scalar_stats["Mean number changes per step: normal"].update(mean_normal_changes)
        scalar_stats["Mean number changes per step: freeze z_L"].update(mean_freeze_changes)
        scalar_stats["Exact accuracy: normal"].update(exact_normal)
        scalar_stats["Exact accuracy: freeze z_L"].update(exact_freeze)

        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} questions...")

    if processed == 0:
        raise RuntimeError("No eval questions were processed.")

    normal_per_step_stats = normal_flip_series.summary()
    freeze_per_step_stats = freeze_flip_series.summary()
    normal_content_stats = normal_content_series.summary()
    freeze_content_stats = freeze_content_series.summary()

    table_headers = ["Metric", "N", "Mean", "Std", "SEM"]
    table_rows = _table_rows_from_stats(metric_order, scalar_stats)
    table_text = markdown_table(table_headers, table_rows)
    write_markdown_table(
        os.path.join(OUT_DIR, "sudoku_freeze_zL_summary.md"),
        table_headers,
        table_rows,
        title="Sudoku Freeze z_L Summary",
    )
    write_json(
        os.path.join(OUT_DIR, "sudoku_freeze_zL_results.json"),
        {
            "num_questions": processed,
            "aggregate": {
                "normal_flips_per_step": {
                    "count": normal_per_step_stats["count"],
                    "mean": normal_per_step_stats["mean"],
                    "std": normal_per_step_stats["std"],
                    "sem": normal_per_step_stats["sem"],
                },
                "freeze_flips_per_step": {
                    "count": freeze_per_step_stats["count"],
                    "mean": freeze_per_step_stats["mean"],
                    "std": freeze_per_step_stats["std"],
                    "sem": freeze_per_step_stats["sem"],
                },
                "normal_number_changes_per_step": {
                    "count": normal_content_stats["count"],
                    "mean": normal_content_stats["mean"],
                    "std": normal_content_stats["std"],
                    "sem": normal_content_stats["sem"],
                },
                "freeze_number_changes_per_step": {
                    "count": freeze_content_stats["count"],
                    "mean": freeze_content_stats["mean"],
                    "std": freeze_content_stats["std"],
                    "sem": freeze_content_stats["sem"],
                },
                "summary_table": table_rows,
            },
        },
    )


    print("Summary table:")
    print(table_text)
    print("Results written to", OUT_DIR)


if __name__ == "__main__":
    main()
