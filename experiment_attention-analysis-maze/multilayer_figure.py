"""Maze multi-layer attention figure: per-cell density bars (delta_rho at radii 4/8/5x5)
on the left y-axis, shape/violation contrast bars (delta_ent, delta_viol) on the right."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


SIZES = {"nbr4": 4, "nbr8": 8, "window5": 24}

DENSITY_METRICS = [
    ("nbr4",    r"$\Delta\rho^{(4)}$",        "#2E5A8A"),
    ("nbr8",    r"$\Delta\rho^{(8)}$",        "#4F88C0"),
    ("window5", r"$\Delta\rho^{(5\times5)}$", "#94BEE0"),
]
SHAPE_METRICS = [
    ("entropy",   r"$\Delta_{\mathrm{ent}}$",  "#59A14F"),
    ("violation", r"$\Delta_{\mathrm{viol}}$", "#F28E2B"),
]
CLASSES = ["error-adjacent", "control"]


def _load_layer_stats(base_dir: str, layer: int):
    path = os.path.join(base_dir, f"quant_summary_layer{layer}.json")
    with open(path) as f:
        payload = json.load(f)
    classes = payload["classes"]
    out = {}
    for cls, d in classes.items():
        out[cls] = {"n_puzzles": int(d["n_puzzles"])}
        for k in ("nbr4", "nbr8", "window5", "entropy", "violation"):
            out[cls][k] = (float(d[k]["mean"]), float(d[k]["ci95"]))
    return out


def build_density_figure(base_dir, output_png, layers=(0,1,2,3), head_mode="avg"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 40,
        "axes.linewidth": 4.5,
        "xtick.major.width": 3.8,
        "ytick.major.width": 3.8,
    })

    stats = {L: _load_layer_stats(base_dir, L) for L in layers}

    fig, axes = plt.subplots(1, len(layers), figsize=(7.5 * len(layers), 8.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.22, wspace=0.10)
    if len(layers) == 1:
        axes = [axes]

    n_m = len(DENSITY_METRICS)
    x = np.arange(len(CLASSES), dtype=np.float64)
    width = 0.75 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width

    # Global y-max so all panels share scale
    all_means = []; all_cis = []
    for L in layers:
        for c in CLASSES:
            for m, _, _ in DENSITY_METRICS:
                mean, ci = stats[L][c][m]
                all_means.append(mean); all_cis.append(ci)
    y_max = max(m + c for m, c in zip(all_means, all_cis)) * 1.15

    for ax_idx, L in enumerate(layers):
        ax = axes[ax_idx]
        for m_i, (metric, label, color) in enumerate(DENSITY_METRICS):
            means = []; cis = []
            for c in CLASSES:
                mean, ci = stats[L][c][metric]
                means.append(mean); cis.append(ci)
            ax.bar(
                x + offsets[m_i], means, width=width*0.90,
                color=color, alpha=0.92, edgecolor="black", linewidth=3.5,
                label=label if ax_idx == 0 else None, zorder=3,
            )
            ax.errorbar(
                x + offsets[m_i], means, yerr=cis,
                fmt="none", ecolor="black", elinewidth=4.8, capsize=8.0, capthick=4.8, zorder=4,
            )
        n_err = stats[L]["error-adjacent"]["n_puzzles"]
        n_ctl = stats[L]["control"]["n_puzzles"]
        ax.set_xticks(x, [f"Err.-adj.\n($n$={n_err})", f"Control\n($n$={n_ctl})"], fontsize=40)
        ax.tick_params(axis="y", labelsize=40)
        ax.set_title(f"Layer {L}", fontsize=40, fontweight="bold", pad=14)
        ax.set_ylim(0, y_max)
        ax.axhline(0.0, color="black", lw=4.0, alpha=0.75, zorder=2)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\Delta\rho^{(k)}$ value", fontsize=40)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, fontsize=40, bbox_to_anchor=(0.52, 1.00))

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    plt.rcdefaults()


def build_shape_figure(base_dir, output_png, layers=(0,1,2,3), head_mode="avg"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 40,
        "axes.linewidth": 4.5,
        "xtick.major.width": 3.8,
        "ytick.major.width": 3.8,
    })

    stats = {L: _load_layer_stats(base_dir, L) for L in layers}

    fig, axes = plt.subplots(1, len(layers), figsize=(7.5 * len(layers), 8.0), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.22, wspace=0.10)
    if len(layers) == 1:
        axes = [axes]

    n_m = len(SHAPE_METRICS)
    x = np.arange(len(CLASSES), dtype=np.float64)
    width = 0.55 / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * width

    for ax_idx, L in enumerate(layers):
        ax = axes[ax_idx]
        for m_i, (metric, label, color) in enumerate(SHAPE_METRICS):
            means = []; cis = []
            for c in CLASSES:
                mean, ci = stats[L][c][metric]
                means.append(mean); cis.append(ci)
            ax.bar(
                x + offsets[m_i], means, width=width*0.88,
                color=color, alpha=0.92, edgecolor="black", linewidth=3.5,
                label=label if ax_idx == 0 else None, zorder=3,
            )
            ax.errorbar(
                x + offsets[m_i], means, yerr=cis,
                fmt="none", ecolor="black", elinewidth=4.8, capsize=8.0, capthick=4.8, zorder=4,
            )
        n_err = stats[L]["error-adjacent"]["n_puzzles"]
        n_ctl = stats[L]["control"]["n_puzzles"]
        ax.set_xticks(x, [f"Err.-adj.\n($n$={n_err})", f"Control\n($n$={n_ctl})"], fontsize=40)
        ax.tick_params(axis="y", labelsize=40)
        ax.set_title(f"Layer {L}", fontsize=40, fontweight="bold", pad=14)
        ax.axhline(0.0, color="black", lw=4.0, alpha=0.75, zorder=2)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\Delta$ value", fontsize=40)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, fontsize=40, bbox_to_anchor=(0.52, 1.00))

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    plt.rcdefaults()


def build_combined_figure(base_dir, output_png, layers=(0,1,2,3), head_mode="avg"):
    """Combined figure: per-panel twin y-axes - left=delta_rho, right=delta-value (entropy/violation).

    Each layer panel shows all 5 metrics at the same class-group on the same
    x-position, with the 3 density bars read from the LEFT y-axis (delta_rho) and
    the 2 shape bars read from the RIGHT y-axis (delta value). Colors match the
    original separate figures.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 40,
        "axes.linewidth": 4.5,
        "xtick.major.width": 3.8,
        "ytick.major.width": 3.8,
    })

    stats = {L: _load_layer_stats(base_dir, L) for L in layers}

    fig, axes_left = plt.subplots(1, len(layers), figsize=(7.5 * len(layers), 8.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.70, bottom=0.22, wspace=0.18)
    if len(layers) == 1:
        axes_left = [axes_left]
    axes_right = [ax.twinx() for ax in axes_left]
    # twinx on a sharey'd row creates independent right-axes per panel; share
    # the right-y across panels manually so all panels use the same delta-value scale.
    for axr in axes_right[1:]:
        axr.sharey(axes_right[0])

    n_total = len(DENSITY_METRICS) + len(SHAPE_METRICS)
    x = np.arange(len(CLASSES), dtype=np.float64)
    width = 0.85 / n_total
    offsets = (np.arange(n_total) - (n_total - 1) / 2.0) * width

    # Global y-max for left (delta_rho) and right (delta-value)
    left_vals = []
    right_vals = []
    for L in layers:
        for c in CLASSES:
            for m, _, _ in DENSITY_METRICS:
                mean, ci = stats[L][c][m]
                left_vals.append(mean + ci)
            for m, _, _ in SHAPE_METRICS:
                mean, ci = stats[L][c][m]
                right_vals.append(mean + ci)
    y_max_left = max(left_vals) * 1.15
    y_max_right = max(right_vals) * 1.15

    for ax_idx, L in enumerate(layers):
        axL = axes_left[ax_idx]
        axR = axes_right[ax_idx]
        # density bars on left axis (first 3 offsets)
        for m_i, (metric, label, color) in enumerate(DENSITY_METRICS):
            means = []; cis = []
            for c in CLASSES:
                mean, ci = stats[L][c][metric]
                means.append(mean); cis.append(ci)
            axL.bar(
                x + offsets[m_i], means, width=width*0.90,
                color=color, alpha=0.92, edgecolor="black", linewidth=3.5,
                label=label if ax_idx == 0 else None, zorder=3,
            )
            axL.errorbar(
                x + offsets[m_i], means, yerr=cis,
                fmt="none", ecolor="black", elinewidth=4.8, capsize=8.0, capthick=4.8, zorder=4,
            )
        # shape bars on right axis (last 2 offsets)
        for j, (metric, label, color) in enumerate(SHAPE_METRICS):
            m_i = len(DENSITY_METRICS) + j
            means = []; cis = []
            for c in CLASSES:
                mean, ci = stats[L][c][metric]
                means.append(mean); cis.append(ci)
            axR.bar(
                x + offsets[m_i], means, width=width*0.90,
                color=color, alpha=0.92, edgecolor="black", linewidth=3.5,
                label=label if ax_idx == 0 else None, zorder=3,
            )
            axR.errorbar(
                x + offsets[m_i], means, yerr=cis,
                fmt="none", ecolor="black", elinewidth=4.8, capsize=8.0, capthick=4.8, zorder=4,
            )
        n_err = stats[L]["error-adjacent"]["n_puzzles"]
        n_ctl = stats[L]["control"]["n_puzzles"]
        axL.set_xticks(x, [f"Err.-adj.\n($n$={n_err})", f"Control\n($n$={n_ctl})"], fontsize=40)
        axL.tick_params(axis="y", labelsize=40)
        axR.tick_params(axis="y", labelsize=40)
        axL.set_title(f"Layer {L}", fontsize=40, fontweight="bold", pad=14)
        axL.set_ylim(0, y_max_left)
        axR.set_ylim(0, y_max_right)
        axL.axhline(0.0, color="black", lw=4.0, alpha=0.75, zorder=2)
        axL.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=1)
        axL.spines["top"].set_visible(False)
        axR.spines["top"].set_visible(False)
        # Hide right tick labels on inner panels (only show on rightmost)
        if ax_idx != len(layers) - 1:
            axR.tick_params(axis="y", labelright=False)

    axes_left[0].set_ylabel(r"$\Delta\rho^{(k)}$ value", fontsize=40)
    axes_right[-1].set_ylabel(r"$\Delta$ value", fontsize=40, rotation=270, labelpad=42)

    handles_L, labels_L = axes_left[0].get_legend_handles_labels()
    handles_R, labels_R = axes_right[0].get_legend_handles_labels()
    fig.legend(handles_L + handles_R, labels_L + labels_R,
               loc="upper center", ncol=5, frameon=False, fontsize=40,
               bbox_to_anchor=(0.50, 1.00))

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    plt.rcdefaults()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="visualization/outputs/maze")
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--density-png", required=False)
    p.add_argument("--shape-png", required=False)
    p.add_argument("--combined-png", required=False)
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    if args.density_png:
        build_density_figure(args.base_dir, args.density_png, layers=layers)
        print(f"Wrote {args.density_png}")
    if args.shape_png:
        build_shape_figure(args.base_dir, args.shape_png, layers=layers)
        print(f"Wrote {args.shape_png}")
    if args.combined_png:
        build_combined_figure(args.base_dir, args.combined_png, layers=layers)
        print(f"Wrote {args.combined_png}")


if __name__ == "__main__":
    main()
