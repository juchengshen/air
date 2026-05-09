"""Sudoku multi-layer attention figure (one panel per layer; three bars per
class - delta_nbr, delta_ent, delta_viol), driven by per-layer head-averaged stats JSONs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


METRICS: List[Tuple[str, str, str]] = [
    ("delta_neighborhood", r"$\Delta_{\mathrm{nbr}}$", "#4E79A7"),
    ("delta_entropy",      r"$\Delta_{\mathrm{ent}}$", "#59A14F"),
    ("delta_violation",    r"$\Delta_{\mathrm{viol}}$", "#F28E2B"),
]
CLASSES = ["violation-adjacent", "control"]


def build(base_dir: str, output_png: str, layers=(0, 1, 2, 3)) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 40,
        "axes.linewidth": 4.5,
        "xtick.major.width": 3.8,
        "ytick.major.width": 3.8,
    })

    stats = {}
    for L in layers:
        with open(os.path.join(base_dir, f"quant_summary_layer{L}.json")) as f:
            stats[L] = json.load(f)

    fig, axes = plt.subplots(1, len(layers), figsize=(7.5 * len(layers), 8.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.22, wspace=0.10)
    if len(layers) == 1:
        axes = [axes]

    n_m = len(METRICS)
    x = np.arange(len(CLASSES), dtype=np.float64)
    width = 0.75 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width

    # global y-range
    all_vals = []
    for L in layers:
        for cls in CLASSES:
            for mkey, _, _ in METRICS:
                m = stats[L]["classes"][cls]["metrics"][mkey]
                all_vals.append(float(m["mean"]) + float(m["ci95"]))
                all_vals.append(float(m["mean"]) - float(m["ci95"]))
    y_min = min(0.0, min(all_vals)) - 0.01
    y_max = max(all_vals) * 1.1

    for ax_idx, L in enumerate(layers):
        ax = axes[ax_idx]
        for m_i, (mkey, mlabel, color) in enumerate(METRICS):
            means = []
            cis = []
            for cls in CLASSES:
                m = stats[L]["classes"][cls]["metrics"][mkey]
                means.append(float(m["mean"]))
                cis.append(float(m["ci95"]))
            ax.bar(
                x + offsets[m_i], means, width=width * 0.90,
                color=color, alpha=0.92, edgecolor="black", linewidth=3.5,
                label=mlabel if ax_idx == 0 else None, zorder=3,
            )
            ax.errorbar(
                x + offsets[m_i], means, yerr=cis,
                fmt="none", ecolor="black", elinewidth=4.8, capsize=8.0, capthick=4.8, zorder=4,
            )
        n_viol = stats[L]["classes"]["violation-adjacent"]["n_puzzles"]
        n_ctrl = stats[L]["classes"]["control"]["n_puzzles"]
        ax.set_xticks(x, [f"Viol.-adj.\n($n$={n_viol})", f"Control\n($n$={n_ctrl})"], fontsize=40)
        ax.tick_params(axis="y", labelsize=40)
        ax.set_title(f"Layer {L}", fontsize=40, fontweight="bold", pad=14)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.0, color="black", lw=4.0, alpha=0.75, zorder=2)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\Delta$ value", fontsize=40)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=3, frameon=False, fontsize=40, bbox_to_anchor=(0.52, 1.00))

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    plt.rcdefaults()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="visualization/outputs")
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--output-png", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    build(args.base_dir, args.output_png, layers=layers)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
