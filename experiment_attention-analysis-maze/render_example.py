#!/usr/bin/env python3
"""Render the Maze attention-analysis example figures from precomputed data.

Reads:
- example_data/core_example.npz           : L/H attention maps, decoded z_H,
                                            label and input grids
- example_data/core_example_index.json    : chosen (puzzle, cycle, head, layer, row, col)
- example_data/temporal_example.npz           : per-cycle L/H maps + grids
- example_data/temporal_example_index.json    : chosen (puzzle, head, layer, row, col, cycles)

Writes maze_core_comparison.png and maze_temporal_evolution.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import plots  # noqa: E402


def render_core(example_dir: str, output_dir: str) -> str:
    with open(os.path.join(example_dir, "core_example_index.json")) as f:
        idx = json.load(f)
    npz = np.load(os.path.join(example_dir, "core_example.npz"))

    out_png = os.path.join(output_dir, "maze_core_comparison.png")
    plots.render_maze_core_comparison(
        out_png,
        l_map=npz["l_map"],
        h_map=npz["h_map"],
        query_idx=int(idx["query_idx"]),
        head_idx=int(idx["head"]),
        layer_idx=int(idx["layer"]),
        act_step=int(idx["cycle"]),
        input_grid=npz["input_grid"],
        label_grid=npz["label_grid"],
        decoded_h_grid=npz["decoded_h_grid"],
        zoom_radius=5,
    )
    return out_png


def render_temporal(example_dir: str, output_dir: str) -> str:
    with open(os.path.join(example_dir, "temporal_example_index.json")) as f:
        idx = json.load(f)
    npz = np.load(os.path.join(example_dir, "temporal_example.npz"))
    cycles = list(npz["cycles"].tolist())

    q = int(idx["query_idx"])
    out_png = os.path.join(output_dir, "maze_temporal_evolution.png")
    plots.render_maze_temporal_grid(
        output_png=out_png,
        steps=cycles,
        l_rows=[npz["l_maps"][i][q] for i in range(len(cycles))],
        h_rows=[npz["h_maps"][i][q] for i in range(len(cycles))],
        query_idx=q,
        head_idx=int(idx["head"]),
        layer_idx=int(idx["layer"]),
        input_grid=npz["input_grid"],
        label_grid=npz["label_grid"],
        decoded_h_grids=[npz["decoded_h_grids"][i] for i in range(len(cycles))],
        zoom_radius=5,
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
