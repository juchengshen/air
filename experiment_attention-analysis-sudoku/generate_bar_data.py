#!/usr/bin/env python3
"""Generate Sudoku attention bar-chart data from a checkpoint.

For each of N test puzzles, captures L/H attention maps at the canonical
sub-step set {2,4,6,8,10,12,14,15} for every (layer, head), scores every
blank query cell against the row/col/box neighborhood and the decoded z_H
violation mask, head-averages over the 8 heads per cell, classifies each
cell as violation-adjacent or control, and aggregates per-class per-layer
mean +/- 95% CI for delta_nbr, delta_ent, delta_viol. Writes one JSON per layer to
--output-dir (default: ./bar_data) in the same shape multilayer_figure.py
consumes.
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

import commons  # noqa: E402  -- local module
from metrics import compute_violation_mask, head_metrics_for_cell  # noqa: E402

CANONICAL_CYCLES: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 15)
NUM_HEADS_DEFAULT: int = 8
NUM_LAYERS_DEFAULT: int = 4


def _per_layer_capture(
    ctx: commons.AttentionContext,
    batch: Dict[str, torch.Tensor],
    *,
    layers: List[int],
    cycles: Tuple[int, ...],
    num_heads: int,
):
    """Return decoded grids per cycle plus
    maps[layer][cycle][head]["L"|"H"] = (81, 81)."""
    decoded = commons.decode_stage_grids_all_steps(
        air_model=ctx.model,
        batch=batch,
        max_act_steps=max(cycles) + 1,
        h_step=0,
        l_step=0,
    )
    maps_by_layer: Dict[int, Dict[int, Dict[int, Dict[str, np.ndarray]]]] = {}
    for layer in layers:
        maps = commons.capture_lh_maps_all_steps(
            air_model=ctx.model,
            batch=batch,
            max_act_steps=max(cycles) + 1,
            h_step=0,
            l_step=0,
            layer_idx=layer,
            head_indices=list(range(num_heads)),
            selected_act_steps=set(cycles),
        )
        maps_by_layer[layer] = maps
    return decoded, maps_by_layer


def _classify_cell(query_idx: int, violation_mask: np.ndarray) -> str:
    return "violation-adjacent" if commons.violation_neighbor_count(query_idx, violation_mask) > 0 else "control"


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute per-class mean / 95% CI / pct_positive for each metric."""
    out: Dict[str, Dict[str, float]] = {}
    by_class: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    n_puzzles_by_class: Dict[str, set] = defaultdict(set)
    for r in rows:
        cls = r["cell_class"]
        n_puzzles_by_class[cls].add(int(r["puzzle_index"]))
        for k in ("delta_neighborhood", "delta_entropy", "delta_violation"):
            by_class[cls][k].append(float(r[k]))

    for cls, metrics in by_class.items():
        cls_block: Dict[str, Dict[str, float]] = {}
        for mname, vals in metrics.items():
            arr = np.asarray(vals, dtype=np.float64)
            n = arr.size
            if n == 0:
                cls_block[mname] = {"mean": 0.0, "ci95": 0.0, "pct_positive": 0.0}
                continue
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
            pct_pos = float(np.mean(arr > 0))
            cls_block[mname] = {"mean": mean, "ci95": ci95, "pct_positive": pct_pos}
        out[cls] = {
            "n_puzzles": len(n_puzzles_by_class[cls]),
            "metrics": cls_block,
        }
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

    ctx = commons.load_checkpoint_and_test_loader(args.ckpt, eval_batch_size=1)
    print(f"Loaded checkpoint: {ctx.checkpoint_path}")

    # Accumulate per-layer rows.
    per_layer_rows: Dict[int, List[Dict[str, float]]] = {L: [] for L in layers}
    eval_iter = iter(ctx.eval_loader)
    for puzzle_index in range(args.num_puzzles):
        try:
            _set_name, batch, _ = next(eval_iter)
        except StopIteration:
            break
        batch = {k: v.to(ctx.device) for k, v in batch.items()}
        inputs_flat = batch["inputs"].detach().cpu().numpy().reshape(-1)

        decoded, maps_by_layer = _per_layer_capture(
            ctx, batch, layers=layers, cycles=cycles, num_heads=args.num_heads
        )

        for cycle in cycles:
            if cycle not in decoded:
                continue
            decoded_h = decoded[cycle]["H"]["pred_H"]
            violation_mask = compute_violation_mask(decoded_h)
            blank_cells = np.where(inputs_flat == commons.BLANK_TOKEN)[0]
            for q in blank_cells.tolist():
                cell_class = _classify_cell(q, violation_mask)
                for layer in layers:
                    if cycle not in maps_by_layer[layer]:
                        continue
                    head_metrics = []
                    for head in range(args.num_heads):
                        slot = maps_by_layer[layer][cycle].get(head)
                        if slot is None or "L" not in slot or "H" not in slot:
                            continue
                        l_row = slot["L"][q]
                        h_row = slot["H"][q]
                        head_metrics.append(head_metrics_for_cell(l_row, h_row, q, violation_mask))
                    if not head_metrics:
                        continue
                    delta_nbr = float(np.mean([m["neighborhood_mass_L"] - m["neighborhood_mass_H"] for m in head_metrics]))
                    delta_ent = float(np.mean([m["entropy_H"] - m["entropy_L"] for m in head_metrics]))
                    delta_viol = float(np.mean([m["violation_mass_L"] - m["violation_mass_H"] for m in head_metrics]))
                    per_layer_rows[layer].append({
                        "puzzle_index": int(puzzle_index),
                        "cycle": int(cycle),
                        "query_idx": int(q),
                        "cell_class": cell_class,
                        "delta_neighborhood": delta_nbr,
                        "delta_entropy": delta_ent,
                        "delta_violation": delta_viol,
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
