#!/usr/bin/env python3
"""Generate Maze attention bar-chart data from a checkpoint.

For each of N test puzzles, captures L/H attention maps at the canonical
sub-step set {2,4,6,8,10,12,14,15} for every (layer, head). For each query
cell on the solution path, head-averages the per-cell statistics over the
8 attention heads, classifies the cell as error-adjacent or control based
on the decoded z_H violation mask in its 4-connected neighborhood, and
aggregates per-class per-layer mean +/- 95% CI for delta_rho at radii 4 / 8 / 5x5,
the entropy contrast delta_ent, and the violation contrast delta_viol. Writes one
JSON per layer to --output-dir (default: ./bar_data) in the same shape
multilayer_figure.py consumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import commons as maze_commons  # noqa: E402
from metrics import (  # noqa: E402
    GRID_SIZE,
    NUM_CELLS,
    compute_maze_violation_mask,
    head_metrics_for_maze_cell,
    neighborhood_indices,
    violation_neighbor_count,
)

CANONICAL_CYCLES: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 15)
NUM_HEADS_DEFAULT: int = 8


def _classify_cell(query_idx: int, violation_mask: np.ndarray) -> str:
    return "error-adjacent" if violation_neighbor_count(query_idx, violation_mask) > 0 else "control"


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    by_class: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    n_puzzles_by_class: Dict[str, set] = defaultdict(set)
    metric_names = ("nbr4", "nbr8", "window5", "entropy", "violation")
    for r in rows:
        cls = r["cell_class"]
        n_puzzles_by_class[cls].add(int(r["puzzle_index"]))
        for k in metric_names:
            by_class[cls][k].append(float(r[k]))

    for cls, metrics in by_class.items():
        cls_block: Dict[str, Dict[str, float]] = {}
        for mname, vals in metrics.items():
            arr = np.asarray(vals, dtype=np.float64)
            n = arr.size
            mean = float(arr.mean()) if n else 0.0
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
            cls_block[mname] = {"mean": mean, "ci95": ci95}
        out[cls] = {"n_puzzles": len(n_puzzles_by_class[cls]), **cls_block}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to checkpoint step file or directory")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "bar_data"))
    parser.add_argument("--num-puzzles", type=int, default=1000)
    parser.add_argument("--cycles", default=",".join(str(c) for c in CANONICAL_CYCLES))
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS_DEFAULT)
    args = parser.parse_args()

    cycles = tuple(int(c) for c in args.cycles.split(",") if c.strip())
    layers = [int(L) for L in args.layers.split(",") if L.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    ctx = maze_commons.load_checkpoint_and_test_loader(args.ckpt, eval_batch_size=1)
    print(f"Loaded checkpoint: {ctx.checkpoint_path}")

    per_layer_rows: Dict[int, List[Dict[str, float]]] = {L: [] for L in layers}
    eval_iter = iter(ctx.eval_loader)
    for puzzle_index in range(args.num_puzzles):
        try:
            _set_name, batch, _ = next(eval_iter)
        except StopIteration:
            break
        batch = {k: v.to(ctx.device) for k, v in batch.items()}
        labels = batch["labels"].detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
        inputs = batch["inputs"].detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)

        decoded = maze_commons.decode_stage_grids_all_steps_maze(
            air_model=ctx.model,
            batch=batch,
            max_act_steps=max(cycles) + 1,
            h_step=0,
            l_step=0,
        )

        # Solution-path cells = label == TOKEN_PATH (2)
        from metrics import TOKEN_PATH  # local import keeps top-level light
        path_cells = np.where(labels.reshape(-1) == TOKEN_PATH)[0]

        for layer in layers:
            maps = maze_commons.capture_lh_maps_all_steps_maze(
                air_model=ctx.model,
                batch=batch,
                max_act_steps=max(cycles) + 1,
                h_step=0,
                l_step=0,
                layer_idx=layer,
                head_indices=list(range(args.num_heads)),
                selected_act_steps=set(cycles),
            )
            for cycle in cycles:
                if cycle not in decoded or cycle not in maps:
                    continue
                decoded_h = decoded[cycle]["H"]["pred_H"]
                violation_mask = compute_maze_violation_mask(decoded_h, labels)
                for q in path_cells.tolist():
                    cell_class = _classify_cell(q, violation_mask)
                    head_metrics = []
                    for head in range(args.num_heads):
                        slot = maps[cycle].get(head)
                        if slot is None or "L" not in slot or "H" not in slot:
                            continue
                        head_metrics.append(
                            head_metrics_for_maze_cell(
                                slot["L"][q], slot["H"][q], q, violation_mask, labels
                            )
                        )
                    if not head_metrics:
                        continue

                    def _avg(key: str) -> float:
                        return float(np.mean([m[key] for m in head_metrics]))

                    nbr4 = (_avg("nbr4_mass_L") - _avg("nbr4_mass_H")) / 4.0
                    nbr8 = (_avg("nbr8_mass_L") - _avg("nbr8_mass_H")) / 8.0
                    win5 = (_avg("window5_mass_L") - _avg("window5_mass_H")) / 24.0
                    ent = _avg("entropy_H") - _avg("entropy_L")
                    viol = _avg("violation_mass_L") - _avg("violation_mass_H")

                    per_layer_rows[layer].append({
                        "puzzle_index": int(puzzle_index),
                        "cycle": int(cycle),
                        "query_idx": int(q),
                        "cell_class": cell_class,
                        "nbr4": nbr4,
                        "nbr8": nbr8,
                        "window5": win5,
                        "entropy": ent,
                        "violation": viol,
                    })
        if (puzzle_index + 1) % 50 == 0:
            print(f"  processed {puzzle_index + 1} puzzles")

    for layer in layers:
        payload = {
            "layer": layer,
            "checkpoint_path": ctx.checkpoint_path,
            "num_puzzles_processed": args.num_puzzles,
            "cycles": list(cycles),
            "classes": _aggregate(per_layer_rows[layer]),
        }
        out_path = os.path.join(args.output_dir, f"quant_summary_layer{layer}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
