#!/usr/bin/env python3
"""
Decode AIR-1net intermediate Sudoku states for the first eval puzzles.

Run from the AIR_code repo root:
  python experiment_visual-sudoku-decoded-freeze/decode_sudoku_intermediate.py
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["figure.titleweight"] = "bold"

PAPER_CELL = 44
BLANK_FILL = (220, 220, 220)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AIR_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "sudoku_decode_first_10")
OUT_DIR = os.environ.get("AIR_SUDOKU_OUT_DIR", DEFAULT_OUT_DIR)
if not os.path.isabs(OUT_DIR):
    OUT_DIR = os.path.join(AIR_ROOT, OUT_DIR)
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
NUM_QUESTIONS = int(os.environ.get("AIR_VISUAL_NUM_QUESTIONS", "10"))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from commons import (  # noqa: E402
    BLANK_TOKEN,
    aggregate_series,
    batch_grid,
    flatten_recursive_traces,
    load_config_model_batches,
    markdown_table,
    metrics_table_rows,
    summarize_scalars,
    write_json,
    write_markdown_table,
)


@lru_cache(maxsize=8)
def get_dejavu_serif(size: int) -> ImageFont.FreeTypeFont:
    font_path = Path(font_manager.findfont("DejaVu Serif"))
    return ImageFont.truetype(str(font_path), size=size)


def token_grid_to_board(grid: np.ndarray) -> list[str]:
    """Convert token ids to paper-renderer board rows; '.' means undecided."""
    grid = np.asarray(grid, dtype=np.int32)
    rows = []
    for row in grid:
        values = []
        for token in row:
            values.append(str(int(token) - 1) if 2 <= int(token) <= 10 else ".")
        rows.append("".join(values))
    return rows


def render_sudoku_board(board: list[str], *, cell: int = PAPER_CELL) -> Image.Image:
    """Render one Sudoku board with the exact visual style used in the paper plots."""
    grid_size = cell * 9
    image = Image.new("RGB", (grid_size, grid_size), "white")
    draw = ImageDraw.Draw(image)
    font = get_dejavu_serif(size=int(cell * 0.73))

    for row_idx, row in enumerate(board):
        for col_idx, value in enumerate(row):
            if value != ".":
                continue
            left = col_idx * cell
            top = row_idx * cell
            draw.rectangle((left, top, left + cell, top + cell), fill=BLANK_FILL)

    for idx in range(10):
        width = 4 if idx % 3 == 0 else 1
        x = idx * cell
        y = idx * cell
        draw.line((x, 0, x, grid_size), fill="black", width=width)
        draw.line((0, y, grid_size, y), fill="black", width=width)

    for row_idx, row in enumerate(board):
        for col_idx, value in enumerate(row):
            if value == ".":
                continue
            left = col_idx * cell
            top = row_idx * cell
            bbox = draw.textbbox((0, 0), value, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = left + (cell - text_w) / 2
            text_y = top + (cell - text_h) / 2 - 2
            draw.text((text_x, text_y), value, font=font, fill="black")

    return image


def plot_sudoku(ax, grid, title: str = "") -> None:
    """Draw a decoded Sudoku board using the paper figure renderer."""
    image = render_sudoku_board(token_grid_to_board(grid))
    ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)


def plot_mean_with_sem(ax, stats: dict, *, label: str, color: str, alpha: float = 0.2):
    x = np.arange(len(stats["mean"]))
    ax.plot(x, stats["mean"], color=color, linewidth=2.0, label=label)
    ax.fill_between(
        x,
        stats["mean"] - stats["sem"],
        stats["mean"] + stats["sem"],
        color=color,
        alpha=alpha,
        linewidth=0,
    )


def grid_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)
    return float((pred == target).mean())


def blank_fraction(grid: np.ndarray) -> float:
    return float((np.asarray(grid) == BLANK_TOKEN).mean())


def final_decoded_grids(question_run: dict) -> tuple[np.ndarray, np.ndarray]:
    final_zH = question_run["act_step_traces"][-1]["logits"].argmax(dim=-1).detach().cpu().numpy().reshape(-1, 9, 9)[0]
    final_zL = question_run["act_step_traces"][-1]["traces"][-1]["pred_L"][0]
    return final_zH, final_zL


def analyze_question(question_run: dict) -> dict:
    labels = batch_grid(question_run["batch"], "labels")
    steps = flatten_recursive_traces(question_run["act_step_traces"])
    zH_accuracy = [grid_accuracy(step["pred_H"], labels) for step in steps]
    zL_accuracy = [grid_accuracy(step["pred_L"], labels) for step in steps]
    zH_blank_fraction = [blank_fraction(step["pred_H"]) for step in steps]
    zL_blank_fraction = [blank_fraction(step["pred_L"]) for step in steps]
    final_zH, final_zL = final_decoded_grids(question_run)

    return {
        "batch_index": question_run["batch_index"],
        "zH_accuracy_per_step": zH_accuracy,
        "zL_accuracy_per_step": zL_accuracy,
        "zH_blank_fraction_per_step": zH_blank_fraction,
        "zL_blank_fraction_per_step": zL_blank_fraction,
        "summary": {
            "mean_zH_accuracy_per_step": float(np.mean(zH_accuracy)) if zH_accuracy else 0.0,
            "mean_zL_accuracy_per_step": float(np.mean(zL_accuracy)) if zL_accuracy else 0.0,
            "mean_zH_blank_fraction_per_step": float(np.mean(zH_blank_fraction)) if zH_blank_fraction else 0.0,
            "mean_zL_blank_fraction_per_step": float(np.mean(zL_blank_fraction)) if zL_blank_fraction else 0.0,
            "final_zH_accuracy": grid_accuracy(final_zH, labels),
            "final_zL_accuracy": grid_accuracy(final_zL, labels),
        },
    }


def save_question_decodes(question_run: dict) -> None:
    batch = question_run["batch"]
    act_step_traces = question_run["act_step_traces"]
    question_index = question_run["batch_index"]
    question_dir = os.path.join(OUT_DIR, f"question_{question_index:02d}")
    os.makedirs(question_dir, exist_ok=True)

    inputs_np = batch_grid(batch, "inputs")
    labels_np = batch_grid(batch, "labels")
    final_zH, final_zL = final_decoded_grids(question_run)

    write_json(
        os.path.join(question_dir, "decoded_traces.json"),
        {
            "batch_index": question_index,
            "input": inputs_np,
            "solution": labels_np,
            "final_zH": final_zH,
            "final_zL": final_zL,
            "act_steps": [
                {
                    "act_step": step_data["act_step"],
                    "recursive_traces": [
                        {
                            "stage": trace["stage"],
                            "H_step": trace["H_step"],
                            "L_step": trace["L_step"],
                            "pred_H": trace["pred_H"][0],
                            "pred_L": trace["pred_L"][0],
                        }
                        for trace in step_data["traces"]
                    ],
                }
                for step_data in act_step_traces
            ],
        },
    )

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_sudoku(axes[0], inputs_np, f"Input (q{question_index})")
    plot_sudoku(axes[1], labels_np, f"Solution (q{question_index})")
    plot_sudoku(axes[2], final_zH, f"Final pred z_H (q{question_index})")
    plt.tight_layout()
    fig.savefig(os.path.join(question_dir, "input_solution_final.png"), dpi=500)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_sudoku(axes[0], inputs_np, f"Input (q{question_index})")
    plot_sudoku(axes[1], labels_np, f"Solution (q{question_index})")
    plot_sudoku(axes[2], final_zL, f"Final pred z_L (q{question_index})")
    plt.tight_layout()
    fig.savefig(os.path.join(question_dir, "input_solution_final_zL.png"), dpi=500)
    plt.close(fig)

    ncols = 4
    for act_idx, step_data in enumerate(act_step_traces):
        traces = step_data["traces"]
        nrows = (len(traces) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for idx, trace in enumerate(traces):
            label = f"{trace['stage']} H{trace['H_step']}"
            if trace["L_step"] is not None:
                label += f" L{trace['L_step']}"
            plot_sudoku(axes[idx], trace["pred_H"][0], label)
        for idx in range(len(traces), len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(f"Question {question_index}, ACT step {act_idx}: decoded from z_H")
        plt.tight_layout()
        fig.savefig(os.path.join(question_dir, f"act{act_idx}_zH_per_recursive_step.png"), dpi=500)
        plt.close(fig)

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for idx, trace in enumerate(traces):
            label = f"{trace['stage']} H{trace['H_step']}"
            if trace["L_step"] is not None:
                label += f" L{trace['L_step']}"
            plot_sudoku(axes[idx], trace["pred_L"][0], label)
        for idx in range(len(traces), len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(f"Question {question_index}, ACT step {act_idx}: decoded from z_L")
        plt.tight_layout()
        fig.savefig(os.path.join(question_dir, f"act{act_idx}_zL_per_recursive_step.png"), dpi=500)
        plt.close(fig)

    nsteps = len(act_step_traces)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    for idx, step_data in enumerate(act_step_traces):
        plot_sudoku(axes[idx], step_data["traces"][-1]["pred_H"][0], f"ACT step {idx}")
    for idx in range(nsteps, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f"Question {question_index}: decoded z_H at end of each ACT step")
    plt.tight_layout()
    fig.savefig(os.path.join(question_dir, "zH_per_act_step.png"), dpi=500)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    for idx, step_data in enumerate(act_step_traces):
        plot_sudoku(axes[idx], step_data["traces"][-1]["pred_L"][0], f"ACT step {idx}")
    for idx in range(nsteps, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f"Question {question_index}: decoded z_L at end of each ACT step")
    plt.tight_layout()
    fig.savefig(os.path.join(question_dir, "zL_per_act_step.png"), dpi=500)
    plt.close(fig)


def main() -> None:
    print(f"Running traced decoding on the first {NUM_QUESTIONS} eval questions...")
    _config, _air, _inner, question_runs = load_config_model_batches(num_questions=NUM_QUESTIONS)
    print(f"Collected {len(question_runs)} questions.")

    question_results = [analyze_question(question_run) for question_run in question_runs]
    zH_accuracy_stats = aggregate_series([result["zH_accuracy_per_step"] for result in question_results])
    zL_accuracy_stats = aggregate_series([result["zL_accuracy_per_step"] for result in question_results])
    zH_blank_stats = aggregate_series([result["zH_blank_fraction_per_step"] for result in question_results])
    zL_blank_stats = aggregate_series([result["zL_blank_fraction_per_step"] for result in question_results])

    table_headers = ["Metric", "N", "Mean", "Std", "SEM"]
    table_rows = metrics_table_rows(
        {
            "Mean z_H accuracy per step": [result["summary"]["mean_zH_accuracy_per_step"] for result in question_results],
            "Mean z_L accuracy per step": [result["summary"]["mean_zL_accuracy_per_step"] for result in question_results],
            "Final z_H accuracy": [result["summary"]["final_zH_accuracy"] for result in question_results],
            "Final z_L accuracy": [result["summary"]["final_zL_accuracy"] for result in question_results],
            "Mean z_H blank fraction per step": [
                result["summary"]["mean_zH_blank_fraction_per_step"] for result in question_results
            ],
            "Mean z_L blank fraction per step": [
                result["summary"]["mean_zL_blank_fraction_per_step"] for result in question_results
            ],
        }
    )
    table_text = markdown_table(table_headers, table_rows)
    write_markdown_table(
        os.path.join(OUT_DIR, "decode_sudoku_intermediate_summary.md"),
        table_headers,
        table_rows,
        title="Decoded Sudoku Intermediate Summary",
    )
    write_json(
        os.path.join(OUT_DIR, "decode_sudoku_intermediate_results.json"),
        {
            "num_questions": len(question_results),
            "per_question": question_results,
            "aggregate": {
                "zH_accuracy_per_step": {
                    "count": zH_accuracy_stats["count"],
                    "mean": zH_accuracy_stats["mean"],
                    "std": zH_accuracy_stats["std"],
                    "sem": zH_accuracy_stats["sem"],
                },
                "zL_accuracy_per_step": {
                    "count": zL_accuracy_stats["count"],
                    "mean": zL_accuracy_stats["mean"],
                    "std": zL_accuracy_stats["std"],
                    "sem": zL_accuracy_stats["sem"],
                },
                "zH_blank_fraction_per_step": {
                    "count": zH_blank_stats["count"],
                    "mean": zH_blank_stats["mean"],
                    "std": zH_blank_stats["std"],
                    "sem": zH_blank_stats["sem"],
                },
                "zL_blank_fraction_per_step": {
                    "count": zL_blank_stats["count"],
                    "mean": zL_blank_stats["mean"],
                    "std": zL_blank_stats["std"],
                    "sem": zL_blank_stats["sem"],
                },
                "summary_table": table_rows,
            },
        },
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_mean_with_sem(axes[0], zH_accuracy_stats, label="z_H", color="steelblue", alpha=0.22)
    plot_mean_with_sem(axes[0], zL_accuracy_stats, label="z_L", color="coral", alpha=0.22)
    axes[0].set_xlabel("Recursive step")
    axes[0].set_ylabel("Cell accuracy")
    axes[0].set_title("Decoded accuracy per step")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    plot_mean_with_sem(axes[1], zH_blank_stats, label="z_H", color="steelblue", alpha=0.22)
    plot_mean_with_sem(axes[1], zL_blank_stats, label="z_L", color="coral", alpha=0.22)
    axes[1].set_xlabel("Recursive step")
    axes[1].set_ylabel("Blank fraction")
    axes[1].set_title("Decoded blank fraction per step")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    final_zH_summary = summarize_scalars([result["summary"]["final_zH_accuracy"] for result in question_results])
    final_zL_summary = summarize_scalars([result["summary"]["final_zL_accuracy"] for result in question_results])
    axes[2].bar(
        ["Final z_H", "Final z_L"],
        [final_zH_summary["mean"], final_zL_summary["mean"]],
        yerr=[final_zH_summary["sem"], final_zL_summary["sem"]],
        capsize=5,
        color=["steelblue", "coral"],
        alpha=0.82,
    )
    axes[2].set_ylabel("Final accuracy")
    axes[2].set_title("Final decoded accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "decode_summary_plots.png"), dpi=500)
    plt.close()

    question_dirs = []
    for question_run in question_runs:
        save_question_decodes(question_run)
        question_dirs.append(os.path.join(OUT_DIR, f"question_{question_run['batch_index']:02d}"))

    print("Summary table:")
    print(table_text)
    print("Done. Outputs saved under:")
    print(" ", OUT_DIR)
    for name in sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".png") or f.endswith(".md") or f.endswith(".json")):
        print(" ", os.path.join(OUT_DIR, name))
    for question_dir in question_dirs:
        print(" ", question_dir)


if __name__ == "__main__":
    main()
