#!/usr/bin/env python3
"""Aggregate per-seed attention-intervention JSONs into the paper tables.

Reads the run_interventions.py outputs for one task, reports mean +/- sample SD
across seeds for per-cell accuracy, exact solve, and (Sudoku) violations per
wrong cell, and runs the seed-level Welch t-tests quoted in the paper. A
one-sided test is used only for a disruption predicted to degrade a metric
relative to baseline; every other comparison is two-sided. Writes a Markdown
summary and a JSON with the same numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# (metric, condition A, condition B, alternative for A vs B)
COMPARISONS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "sudoku": [
        ("cell_accuracy", "L_diffuse", "baseline", "less"),
        ("cell_accuracy", "L_excludes_neighborhood", "baseline", "less"),
        ("cell_accuracy", "L_local_only", "L_random_only", "two-sided"),
        ("violations_per_wrong_cell", "L_diffuse", "baseline", "greater"),
        ("violations_per_wrong_cell", "H_diffuse", "baseline", "two-sided"),
    ],
    "maze": [
        ("cell_accuracy", "L_diffuse", "baseline", "less"),
        ("cell_accuracy", "L_excludes_neighborhood", "baseline", "less"),
        ("cell_accuracy", "L_local_only", "L_random_only", "two-sided"),
        ("exact_accuracy", "H_local_only", "baseline", "less"),
        ("cell_accuracy", "H_local_only", "baseline", "two-sided"),
        ("exact_accuracy", "H_diffuse", "baseline", "two-sided"),
        ("exact_accuracy", "H_excludes_neighborhood", "baseline", "two-sided"),
        ("cell_accuracy", "L_local_H_diffuse", "L_diffuse_H_local", "two-sided"),
    ],
}


def load_runs(paths: List[str]) -> Tuple[str, List[Dict[str, Dict[str, float]]]]:
    """Load per-seed results and add violations per wrong cell where available."""
    task = None
    runs = []
    for path in paths:
        with open(path, "r") as f:
            payload = json.load(f)
        if task is not None and payload["task"] != task:
            raise ValueError(f"Mixed tasks in inputs: {task} and {payload['task']}")
        task = payload["task"]
        results = payload["results"]
        for metrics in results.values():
            if "violating_cells" in metrics:
                metrics["violations_per_wrong_cell"] = metrics["violating_cells"] / max(1, metrics["wrong_cells"])
        runs.append(results)
    if task is None:
        raise ValueError("No result files found.")
    return task, runs


def seed_values(runs: List[Dict[str, Dict[str, float]]], condition: str, metric: str) -> np.ndarray:
    return np.asarray([run[condition][metric] for run in runs], dtype=np.float64)


def mean_sd(values: np.ndarray) -> Tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="glob matching the per-seed JSONs of one task")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.inputs))
    task, runs = load_runs(paths)
    conditions = [c for c in runs[0] if all(c in run for run in runs)]
    metrics = ["cell_accuracy", "exact_accuracy"]
    if task == "sudoku":
        metrics.append("violations_per_wrong_cell")

    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for condition in conditions:
        table[condition] = {}
        for metric in metrics:
            mean, sd = mean_sd(seed_values(runs, condition, metric))
            table[condition][metric] = {"mean": mean, "sd": sd}

    tests = []
    for metric, cond_a, cond_b, alternative in COMPARISONS[task]:
        if cond_a not in conditions or cond_b not in conditions:
            continue
        a = seed_values(runs, cond_a, metric)
        b = seed_values(runs, cond_b, metric)
        result = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
        tests.append({
            "metric": metric,
            "condition_a": cond_a,
            "condition_b": cond_b,
            "alternative": alternative,
            "difference": float(a.mean() - b.mean()),
            "p_value": float(result.pvalue),
        })

    lines = [
        f"# {task.capitalize()} attention interventions",
        "",
        f"{len(runs)} checkpoints, mean +/- sample SD across seeds. Inputs: {', '.join(os.path.basename(p) for p in paths)}",
        "",
        "| Condition | " + " | ".join(metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for condition in conditions:
        cells = []
        for metric in metrics:
            entry = table[condition][metric]
            digits = 2 if metric == "violations_per_wrong_cell" else 1
            cells.append(f"{entry['mean']:.{digits}f} +/- {entry['sd']:.{digits}f}")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    lines += ["", "## Welch t-tests (seed level)", "", "| Metric | A | B | Test | A - B | p |", "|---|---|---|---|---:|---:|"]
    for t in tests:
        lines.append(
            f"| {t['metric']} | {t['condition_a']} | {t['condition_b']} | {t['alternative']} "
            f"| {t['difference']:+.2f} | {t['p_value']:.2g} |"
        )

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.output_md}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"task": task, "inputs": paths, "table": table, "tests": tests}, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
