#!/usr/bin/env python3
"""Plot Maze freeze experiment JSON outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["figure.titleweight"] = "bold"


def infer_freeze_axis(path: Path, data: dict) -> str:
    name = path.name
    if "zH" in name:
        return "z_H"
    if "zL" in name:
        return "z_L"

    text = json.dumps(data.get("metrics", data.get("aggregate", {})))
    if "freeze z_H" in text:
        return "z_H"
    if "freeze z_L" in text:
        return "z_L"
    raise ValueError(f"Could not infer freeze axis from {path}")


def output_path_for(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_results"):
        stem = stem[: -len("_results")] + "_plots"
    return input_path.with_name(f"{stem}.png")


def as_float(value: object) -> float:
    return float(value)


def summary_row(summary_table: list[list[object]], metric_name: str) -> dict[str, float]:
    for row in summary_table:
        if row and row[0] == metric_name:
            return {
                "n": as_float(row[1]),
                "mean": as_float(row[2]),
                "std": as_float(row[3]),
                "sem": as_float(row[4]),
            }
    raise KeyError(f"Metric not found in summary_table: {metric_name}")


def plot_mean_with_std(ax, stats: dict, *, label: str, color: str, alpha: float = 0.22):
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    x = np.arange(mean.size)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def plot_result_json(input_path: Path, data: dict, output_path: Path):
    freeze_axis = infer_freeze_axis(input_path, data)
    freeze_label = f"Freeze {freeze_axis}"
    aggregate = data["aggregate"]
    summary_table = aggregate["summary_table"]

    total_normal = summary_row(summary_table, "Total color changes: normal")
    total_freeze = summary_row(summary_table, f"Total color changes: freeze {freeze_axis}")
    normal_label = "Normal\n(z_H + z_L)" if freeze_axis == "z_H" else "Normal\n(decoded z_H)"
    freeze_bar_label = "Freeze z_H\n(z_L only)" if freeze_axis == "z_H" else "Freeze z_L\n(z_H only)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.flatten()

    axes[0].bar(
        [normal_label, freeze_bar_label],
        [total_normal["mean"], total_freeze["mean"]],
        yerr=[total_normal["std"], total_freeze["std"]],
        capsize=5,
        color=["steelblue", "coral"],
        alpha=0.82,
    )
    axes[0].set_ylabel("Total color changes")
    axes[0].set_title("Total color changes: mean +/- std")

    plot_mean_with_std(axes[1], aggregate["normal_color_changes_per_step"], label="Normal", color="steelblue")
    plot_mean_with_std(
        axes[1],
        aggregate["freeze_color_changes_per_step"],
        label=freeze_label,
        color="coral",
    )
    axes[1].set_xlabel("Recursive step")
    axes[1].set_ylabel("Average color changes")
    axes[1].set_title("Color changes per step (mean +/- std)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()


def metric(data: dict, name: str) -> dict:
    try:
        return data["metrics"][name]
    except KeyError as exc:
        raise KeyError(f"Metric not found in aggregate JSON: {name}") from exc


def plot_metric_bar(ax, data: dict, *, normal_name: str, freeze_name: str, ylabel: str, title: str):
    normal = metric(data, normal_name)
    freeze = metric(data, freeze_name)
    runs = list(data.get("runs", normal["per_run_means"].keys()))
    x = np.arange(2)
    ax.bar(
        x,
        [normal["mean_across_runs"], freeze["mean_across_runs"]],
        yerr=[normal["std_across_runs"], freeze["std_across_runs"]],
        capsize=5,
        color=["steelblue", "coral"],
        alpha=0.82,
    )
    for run in runs:
        if run in normal["per_run_means"] and run in freeze["per_run_means"]:
            values = [normal["per_run_means"][run], freeze["per_run_means"][run]]
            ax.plot(x, values, color="black", linewidth=0.8, alpha=0.35)
            ax.scatter(x, values, color="black", s=12, alpha=0.55)
    ax.set_xticks(x, ["Normal", freeze_name.replace("Total color changes: ", "").replace("Mean color changes per step: ", "")])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_aggregate_json(input_path: Path, data: dict, output_path: Path):
    freeze_axis = infer_freeze_axis(input_path, data)
    freeze_label = f"freeze {freeze_axis}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.flatten()
    plot_metric_bar(
        axes[0],
        data,
        normal_name="Total color changes: normal",
        freeze_name=f"Total color changes: {freeze_label}",
        ylabel="Total color changes",
        title="Total color changes across runs",
    )
    plot_metric_bar(
        axes[1],
        data,
        normal_name="Mean color changes per step: normal",
        freeze_name=f"Mean color changes per step: {freeze_label}",
        ylabel="Mean color changes per step",
        title="Mean color changes per step across runs",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.close()


def plot_json(input_path: Path, output_path: Path | None = None) -> Path:
    with input_path.open("r") as f:
        data = json.load(f)

    output_path = output_path or output_path_for(input_path)
    if "metrics" in data:
        plot_aggregate_json(input_path, data, output_path)
    else:
        plot_result_json(input_path, data, output_path)
    return output_path


def load_metric_means(path: Path) -> dict[str, float]:
    with path.open("r") as f:
        data = json.load(f)
    table = data["aggregate"]["summary_table"]
    return {str(row[0]): float(row[2]) for row in table}


def summarize(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def write_aggregate_from_manifest(manifest_path: Path, *, make_plots: bool) -> list[Path]:
    rows: list[tuple[str, Path]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run_label, out_dir = line.split("|", 1)
            rows.append((run_label, Path(out_dir)))

    artifacts = [
        ("maze_freeze_zH_results.json", "maze_freeze_zH_aggregate_5runs"),
        ("maze_freeze_zL_results.json", "maze_freeze_zL_aggregate_5runs"),
    ]

    written: list[Path] = []
    for result_name, out_prefix in artifacts:
        run_metric_map = {}
        for run_label, out_dir in rows:
            result_path = out_dir / result_name
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing results file: {result_path}")
            run_metric_map[run_label] = load_metric_means(result_path)

        metric_names = list(next(iter(run_metric_map.values())).keys())
        aggregate = {}
        for metric in metric_names:
            vals = [run_metric_map[run_label][metric] for run_label, _ in rows]
            mean, std = summarize(vals)
            aggregate[metric] = {
                "n_runs": len(vals),
                "mean_across_runs": mean,
                "std_across_runs": std,
                "per_run_means": {run_label: run_metric_map[run_label][metric] for run_label, _ in rows},
            }

        root_dir = rows[0][1].parent if rows else Path(".")
        json_path = root_dir / f"{out_prefix}.json"
        md_path = root_dir / f"{out_prefix}.md"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump({"runs": [r for r, _ in rows], "metrics": aggregate}, f, indent=2)

        lines = [
            f"# {out_prefix}",
            "",
            "| Metric | N runs | Mean across runs | Std across runs |",
            "| --- | --- | --- | --- |",
        ]
        for metric, stats in aggregate.items():
            lines.append(
                f"| {metric} | {stats['n_runs']} | {stats['mean_across_runs']:.4f} | {stats['std_across_runs']:.4f} |"
            )
        lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")

        written.extend([json_path, md_path])
        if make_plots:
            written.append(plot_json(json_path))

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="*", type=Path, help="Maze freeze result or aggregate JSON files")
    parser.add_argument("--output", type=Path, help="Output PNG path; only valid with one input JSON")
    parser.add_argument("--aggregate-manifest", type=Path, help="Manifest with run_label|output_dir rows for 5-run aggregation")
    parser.add_argument("--no-plot", action="store_true", help="With --aggregate-manifest, write JSON/Markdown only")
    args = parser.parse_args()

    if args.aggregate_manifest:
        if args.json_paths:
            parser.error("json_paths cannot be used with --aggregate-manifest")
        for output_path in write_aggregate_from_manifest(args.aggregate_manifest, make_plots=not args.no_plot):
            print(f"Wrote {output_path}")
        return

    if args.output and len(args.json_paths) != 1:
        parser.error("--output can only be used with one input JSON")
    if not args.json_paths:
        parser.error("json_paths are required unless --aggregate-manifest is used")

    for json_path in args.json_paths:
        output_path = plot_json(json_path, args.output)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
