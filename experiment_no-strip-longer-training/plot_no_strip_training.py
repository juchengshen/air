#!/usr/bin/env python3
"""Plot longer training of the Lx_Hx prepend (no strip) level-token control.

Reads example_data/no_strip_training_data.json, which holds five-seed mean and SD
at every 2,000-epoch evaluation up to 40,000 epochs for:
- training and evaluation loss,
- training and evaluation exact solve,
- cross-puzzle consistency of the persistent z_H and z_L level-token slots.

Writes no_strip_training.pdf and no_strip_training.png next to the data, or to
the directory passed via --output-dir.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "example_data" / "no_strip_training_data.json")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "example_data")
    args = parser.parse_args()

    data = json.loads(args.data.read_text())
    x = np.asarray(data["epoch_k"])
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "mathtext.fontset": "stix",
        "font.size": 16, "axes.labelsize": 17, "axes.titlesize": 18,
        "legend.fontsize": 14, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), layout="constrained")
    panels = [
        ("(a) Training and evaluation loss", "LM loss", [("train_loss", "Train"), ("eval_loss", "Evaluation")], (0, .95)),
        ("(b) Exact solve", "Exact solve (%)", [("train_exact", "Train"), ("eval_exact", "Evaluation")], (0, 105)),
        ("(c) Persistent-slot consistency", "Mean cosine similarity", [("consistency_H", r"$z_H$ slot"), ("consistency_L", r"$z_L$ slot")], (.45, 1.02)),
    ]
    for ax, (title, ylabel, series, ylim) in zip(axes, panels):
        for (key, label), color in zip(series, ["steelblue", "coral"]):
            mean = np.asarray(data[key]["mean"])
            sd = np.asarray(data[key]["sd"])
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.20, linewidth=0)
            ax.plot(x, mean, color=color, marker="o", markersize=4, linewidth=2.2, label=label)
        ax.axvline(20, color=".5", linewidth=1, linestyle="--", zorder=0)
        ax.set(title=title, xlabel="Training epochs (thousands)", ylabel=ylabel, xlim=(1, 41), ylim=ylim)
        ax.set_xticks([2, 10, 20, 30, 40])
        ax.grid(alpha=.2)
        ax.legend(frameon=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "no_strip_training.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "no_strip_training.png", dpi=220, bbox_inches="tight")
    print(f"wrote {args.output_dir / 'no_strip_training.pdf'}")


if __name__ == "__main__":
    main()
