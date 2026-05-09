#!/usr/bin/env python3
"""
Freeze z_L ablation for Maze.

If we stop updating z_L and let only z_H recurse, does the solution-path mask
still shift?

This tests whether mask motion can still be driven by H dynamics when L is held fixed.

Run from AIR_code root: python experiment_visual-maze-decoded-freeze/maze_freeze_zL.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from commons import (
    DEFAULT_NUM_QUESTIONS,
    SOLUTION_TOKEN,
    aggregate_series,
    batch_grid,
    flatten_recursive_traces,
    load_config_model_batches,
    markdown_table,
    metrics_table_rows,
    write_json,
    write_markdown_table,
)

DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "test_maze_L2x_Hx_20k_epochs")
OUT_DIR = os.environ.get("AIR_MAZE_TEST_OUT_DIR", DEFAULT_OUT_DIR)
if not os.path.isabs(OUT_DIR):
    OUT_DIR = os.path.join(SCRIPT_DIR, OUT_DIR)
OUT_DIR = os.path.abspath(OUT_DIR)
NUM_QUESTIONS = int(os.environ.get("AIR_MAZE_NUM_QUESTIONS", str(DEFAULT_NUM_QUESTIONS)))
os.makedirs(OUT_DIR, exist_ok=True)


def mask_solution(grid: np.ndarray) -> np.ndarray:
    """True where cell is predicted as solution path (token 5). grid shape (30,30)."""
    return (grid == SOLUTION_TOKEN).astype(np.uint8)


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
        transition_metrics(prev_grid, curr_grid, mask_solution)["mask_flips"]
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


def analyze_question_pair(normal_run: dict, freeze_run: dict) -> dict:
    normal_steps = [step["pred_H"] for step in flatten_recursive_traces(normal_run["act_step_traces"])]
    freeze_steps = [step["pred_H"] for step in flatten_recursive_traces(freeze_run["act_step_traces"])]
    labels = batch_grid(normal_run["batch"], "labels")
    per_step_normal = flips_per_step_from_steps(normal_steps)
    per_step_freeze = flips_per_step_from_steps(freeze_steps)
    content_normal = content_changes_per_step_from_steps(normal_steps, mask_solution)
    content_freeze = content_changes_per_step_from_steps(freeze_steps, mask_solution)
    final_normal = normal_steps[-1] if normal_steps else labels
    final_freeze = freeze_steps[-1] if freeze_steps else labels
    return {
        "batch_index": normal_run["batch_index"],
        "normal": {
            "n_steps": len(normal_steps),
            "per_step_flips": per_step_normal,
            "per_step_content_changes": content_normal,
            "total_mask_flips": int(sum(per_step_normal)),
            "total_color_changes": int(sum(content_normal)),
            "mean_flips_per_step": float(np.mean(per_step_normal)) if per_step_normal else 0.0,
            "mean_color_changes_per_step": float(np.mean(content_normal)) if content_normal else 0.0,
            "exact_accuracy": exact_match_accuracy(final_normal, labels),
        },
        "freeze_zL": {
            "n_steps": len(freeze_steps),
            "per_step_flips": per_step_freeze,
            "per_step_content_changes": content_freeze,
            "total_mask_flips": int(sum(per_step_freeze)),
            "total_color_changes": int(sum(content_freeze)),
            "mean_flips_per_step": float(np.mean(per_step_freeze)) if per_step_freeze else 0.0,
            "mean_color_changes_per_step": float(np.mean(content_freeze)) if content_freeze else 0.0,
            "exact_accuracy": exact_match_accuracy(final_freeze, labels),
        },
    }


def main():
    print(f"(1) Normal run across first {NUM_QUESTIONS} eval questions")
    config, air, inner, normal_runs = load_config_model_batches(num_questions=NUM_QUESTIONS)

    print(f"(2) Freeze z_L run across first {NUM_QUESTIONS} eval questions")
    config, air, inner, freeze_runs = load_config_model_batches(
        num_questions=NUM_QUESTIONS,
        freeze_z_L=True,
    )

    paired_results = [
        analyze_question_pair(normal_run, freeze_run)
        for normal_run, freeze_run in zip(normal_runs, freeze_runs)
    ]
    normal_per_step_stats = aggregate_series(
        [result["normal"]["per_step_flips"] for result in paired_results]
    )
    freeze_per_step_stats = aggregate_series(
        [result["freeze_zL"]["per_step_flips"] for result in paired_results]
    )
    normal_content_stats = aggregate_series(
        [result["normal"]["per_step_content_changes"] for result in paired_results]
    )
    freeze_content_stats = aggregate_series(
        [result["freeze_zL"]["per_step_content_changes"] for result in paired_results]
    )

    table_headers = ["Metric", "N", "Mean", "Std", "SEM"]
    table_rows = metrics_table_rows(
        {
            "Total solution-mask flips: normal": [
                result["normal"]["total_mask_flips"] for result in paired_results
            ],
            "Total solution-mask flips: freeze z_L": [
                result["freeze_zL"]["total_mask_flips"] for result in paired_results
            ],
            "Delta total flips (freeze - normal)": [
                result["freeze_zL"]["total_mask_flips"] - result["normal"]["total_mask_flips"]
                for result in paired_results
            ],
            "Total color changes: normal": [result["normal"]["total_color_changes"] for result in paired_results],
            "Total color changes: freeze z_L": [result["freeze_zL"]["total_color_changes"] for result in paired_results],
            "Delta total color changes (freeze - normal)": [
                result["freeze_zL"]["total_color_changes"] - result["normal"]["total_color_changes"]
                for result in paired_results
            ],
            "Mean flips per step: normal": [result["normal"]["mean_flips_per_step"] for result in paired_results],
            "Mean flips per step: freeze z_L": [
                result["freeze_zL"]["mean_flips_per_step"] for result in paired_results
            ],
            "Mean color changes per step: normal": [result["normal"]["mean_color_changes_per_step"] for result in paired_results],
            "Mean color changes per step: freeze z_L": [
                result["freeze_zL"]["mean_color_changes_per_step"] for result in paired_results
            ],
            "Exact accuracy: normal": [result["normal"]["exact_accuracy"] for result in paired_results],
            "Exact accuracy: freeze z_L": [result["freeze_zL"]["exact_accuracy"] for result in paired_results],
        }
    )
    table_text = markdown_table(table_headers, table_rows)
    write_markdown_table(
        os.path.join(OUT_DIR, "maze_freeze_zL_summary.md"),
        table_headers,
        table_rows,
        title="Maze Freeze z_L Summary",
    )
    write_json(
        os.path.join(OUT_DIR, "maze_freeze_zL_results.json"),
        {
            "num_questions": len(paired_results),
            "per_question": paired_results,
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
                "normal_color_changes_per_step": {
                    "count": normal_content_stats["count"],
                    "mean": normal_content_stats["mean"],
                    "std": normal_content_stats["std"],
                    "sem": normal_content_stats["sem"],
                },
                "freeze_color_changes_per_step": {
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
