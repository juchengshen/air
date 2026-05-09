#!/usr/bin/env python3
"""Render the Sudoku attention-analysis example figures from precomputed data.

Reads:
- example_data/core_example.npz       : L/H attention maps + decoded z_H grid
- example_data/core_example_index.json: chosen (puzzle, cycle, head, layer, query)
- example_data/temporal_example.npz       : per-cycle L/H maps + decoded grids
- example_data/temporal_example_index.json: chosen (puzzle, head, layer, query, cycles)

Writes core_comparison.png and temporal_evolution.png next to the data, or to
the directory passed via --output-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import plots  # noqa: E402  -- local module
from metrics import compute_violation_mask, head_metrics_for_cell  # noqa: E402


def _metrics_dict(metrics_payload: Dict[str, Any], side: str) -> Dict[str, float]:
    return {
        "entropy": float(metrics_payload.get(f"entropy_{side}", 0.0)),
        "nbr": float(metrics_payload.get(f"neighborhood_mass_{side}", 0.0)),
        "viol": float(metrics_payload.get(f"violation_mass_{side}", 0.0)),
    }


def render_core(example_dir: str, output_dir: str) -> str:
    with open(os.path.join(example_dir, "core_example_index.json")) as f:
        idx = json.load(f)
    npz = np.load(os.path.join(example_dir, "core_example.npz"))
    l_map = npz["l_map"]
    h_map = npz["h_map"]
    decoded_h = npz["decoded_h_grid_tokens"]

    q = int(idx["query_idx"])
    metrics_payload = head_metrics_for_cell(l_map[q], h_map[q], q, compute_violation_mask(decoded_h))
    out_png = os.path.join(output_dir, "core_comparison.png")
    plots.render_core_comparison(
        output_png=out_png,
        l_map=l_map,
        h_map=h_map,
        query_idx=q,
        head_idx=int(idx["head"]),
        layer_idx=int(idx["layer"]),
        act_step=int(idx["cycle"]),
        decoded_h_grid_tokens=decoded_h,
        solution_grid_tokens=None,
        metrics_l=_metrics_dict(metrics_payload, "L"),
        metrics_h=_metrics_dict(metrics_payload, "H"),
        title_pad=24,
        show_topk_labels=False,
        show_metrics_box=False,
    )
    return out_png


def render_temporal(example_dir: str, output_dir: str) -> str:
    with open(os.path.join(example_dir, "temporal_example_index.json")) as f:
        idx = json.load(f)
    npz = np.load(os.path.join(example_dir, "temporal_example.npz"))
    l_maps = npz["l_maps"]
    h_maps = npz["h_maps"]
    decoded_h_grids = npz["decoded_h_grids_tokens"]
    cycles = list(npz["cycles"].tolist())

    out_png = os.path.join(output_dir, "temporal_evolution.png")
    plots.render_temporal_grid(
        output_png=out_png,
        steps=cycles,
        l_rows=[l_maps[i] for i in range(len(cycles))],
        h_rows=[h_maps[i] for i in range(len(cycles))],
        query_idx=int(idx["query_idx"]),
        head_idx=int(idx["head"]),
        layer_idx=int(idx["layer"]),
        decoded_h_grids_tokens=[decoded_h_grids[i] for i in range(len(cycles))],
    )
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", default=os.path.join(SCRIPT_DIR, "example_data"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.example_dir
    os.makedirs(out_dir, exist_ok=True)

    core_png = render_core(args.example_dir, out_dir)
    print(f"Wrote {core_png}")
    temp_png = render_temporal(args.example_dir, out_dir)
    print(f"Wrote {temp_png}")


if __name__ == "__main__":
    main()
