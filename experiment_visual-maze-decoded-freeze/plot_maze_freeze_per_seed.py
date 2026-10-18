#!/usr/bin/env python3
"""Per-seed Maze freeze figure (paper appendix, per-seed freeze results).

Reads a run manifest written by a maze_freeze_zH_zL_*_3runs.sh script (rows of
run_label|output_dir, with run labels run1, run2, ... for seeds 0, 1, ...) and the
maze_freeze_{zH,zL}_results.json files in each output directory.

Writes one figure with one row per seed and four columns: freeze z_H total,
freeze z_H per sub-step, freeze z_L total, freeze z_L per sub-step. Bars and bands
are mean +/- SD over the evaluated puzzles of that seed. Also prints the per-seed
totals reported in the appendix table.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK = "maze"
KIND = "color"
BLUE, ORANGE = "#4E79A7", "#F28E2B"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42})


def read_manifest(path: Path) -> list[tuple[str, Path]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            run_label, out_dir = line.split("|", 1)
            rows.append((run_label, Path(out_dir)))
    return rows


def seed_of(run_label: str) -> int:
    return int(run_label.removeprefix("run")) - 1


def load(out_dir: Path, fz: str) -> dict:
    data = json.loads((out_dir / f"{TASK}_freeze_{fz}_results.json").read_text())["aggregate"]
    sc = data["scalar_summaries"]
    return dict(
        tot_n=sc[f"Total {KIND} changes: normal"],
        tot_f=sc[f"Total {KIND} changes: freeze z_{fz[1]}"],
        step_n=data[f"normal_{KIND}_changes_per_step"],
        step_f=data[f"freeze_{KIND}_changes_per_step"],
    )


def bar(ax, r, fz):
    ax.bar([0, 1], [r["tot_n"]["mean"], r["tot_f"]["mean"]], yerr=[r["tot_n"]["std"], r["tot_f"]["std"]],
           color=[BLUE, ORANGE], edgecolor="black", linewidth=0.5, width=0.75, capsize=2,
           error_kw=dict(elinewidth=0.6, capthick=0.6))
    ax.set_xticks([0, 1], ["Normal", f"Freeze $z_{fz[1]}$"])
    ax.set_ylabel("Total")


def curve(ax, r):
    for s, c in [(r["step_n"], BLUE), (r["step_f"], ORANGE)]:
        m = np.asarray(s["mean"], float)
        sd = np.asarray(s["std"], float)
        x = np.arange(m.size)
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.18, linewidth=0)
        ax.plot(x, m, color=c, linewidth=0.9)
    ax.set_ylabel("Per sub-step")
    ax.set_xlabel("Sub-step")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="run_manifest.txt written by a 3-run freeze script")
    parser.add_argument("--output", type=Path, help="Output PDF path (a PNG is written alongside)")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    output = args.output or args.manifest.parent / f"{args.manifest.parent.name}_per_seed.pdf"

    fig, axes = plt.subplots(len(rows), 4, figsize=(6.75, 1.45 * len(rows) + 0.35),
                             gridspec_kw=dict(width_ratios=[1, 1.5, 1, 1.5]), squeeze=False)
    totals = []
    for i, (run_label, out_dir) in enumerate(rows):
        for j, fz in enumerate(["zH", "zL"]):
            r = load(out_dir, fz)
            bar(axes[i, 2 * j], r, fz)
            curve(axes[i, 2 * j + 1], r)
            totals.append((seed_of(run_label), fz, r["tot_n"]["mean"], r["tot_f"]["mean"]))
        axes[i, 0].set_ylabel(f"Seed {seed_of(run_label)}\nTotal", fontweight="bold")
    axes[0, 0].set_title("Freeze $z_H$: total", fontsize=7.5)
    axes[0, 1].set_title("Freeze $z_H$: per sub-step", fontsize=7.5)
    axes[0, 2].set_title("Freeze $z_L$: total", fontsize=7.5)
    axes[0, 3].set_title("Freeze $z_L$: per sub-step", fontsize=7.5)
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    for ax in axes[:-1, [1, 3]].flat:
        ax.set_xlabel("")
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {output}")

    print(f"\nPer-seed total {KIND} changes (mean over evaluated puzzles)")
    for seed, fz, normal, frozen in totals:
        print(f"seed {seed}  freeze_{fz}  normal={normal:9.1f}  frozen={frozen:9.1f}  delta={frozen - normal:+9.1f}")


if __name__ == "__main__":
    main()
