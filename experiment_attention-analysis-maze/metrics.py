"""Maze L/H attention metrics.

Per-cell + per-head L/H attention metrics for a 30x30 maze grid with three
candidate neighborhood definitions (4-connected, 8-connected, 5x5 window).
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from numpy.typing import NDArray


GRID_SIZE = 30
NUM_CELLS = GRID_SIZE * GRID_SIZE

# Token ids from dataset/build_maze_dataset.py
TOKEN_PAD = 0
TOKEN_WALL = 1
TOKEN_SPACE = 2
TOKEN_START = 3
TOKEN_GOAL = 4
TOKEN_PATH = 5


def _validate_query_idx(query_idx: int) -> int:
    q = int(query_idx)
    if q < 0 or q >= NUM_CELLS:
        raise ValueError(f"query_idx out of range [0, {NUM_CELLS - 1}]: {query_idx}")
    return q


def _as_grid_int(values: np.ndarray) -> NDArray[np.int64]:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size != NUM_CELLS:
        raise ValueError(f"expected {NUM_CELLS} values for a {GRID_SIZE}x{GRID_SIZE} grid, got {arr.size}")
    return arr.reshape(GRID_SIZE, GRID_SIZE)


def _as_grid_bool(values: np.ndarray) -> NDArray[np.bool_]:
    arr = np.asarray(values, dtype=bool)
    if arr.size != NUM_CELLS:
        raise ValueError(f"expected {NUM_CELLS} values for a {GRID_SIZE}x{GRID_SIZE} grid, got {arr.size}")
    return arr.reshape(GRID_SIZE, GRID_SIZE)


def _safe_prob_row(attn_row: np.ndarray) -> NDArray[np.float64]:
    row = np.asarray(attn_row, dtype=np.float64).reshape(-1)
    if row.size != NUM_CELLS:
        raise ValueError(f"attention row must contain {NUM_CELLS} values, got {row.size}")
    row = np.clip(row, 0.0, None)
    denom = float(row.sum())
    if denom <= 0.0:
        return np.full(NUM_CELLS, 1.0 / NUM_CELLS, dtype=np.float64)
    return row / denom


def neighborhood_indices(query_idx: int, kind: str = "4conn") -> NDArray[np.int64]:
    """Return flat indices of the neighborhood (excluding self).

    kind:
      - "4conn": up/down/left/right neighbors (up to 4)
      - "8conn": up/down/left/right + diagonals (up to 8)
      - "window5": all cells in a 5x5 window centered on q (up to 24)
    """
    q = _validate_query_idx(query_idx)
    r, c = divmod(q, GRID_SIZE)

    if kind == "4conn":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif kind == "8conn":
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    elif kind == "window5":
        offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)]
    else:
        raise ValueError(f"unknown neighborhood kind: {kind}")

    idxs = []
    for dr, dc in offsets:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
            idxs.append(rr * GRID_SIZE + cc)
    return np.asarray(sorted(idxs), dtype=np.int64)


def compute_maze_violation_mask(
    decoded_grid: np.ndarray, label_grid: np.ndarray
) -> NDArray[np.bool_]:
    """Boolean (30, 30) mask marking cells where decoded != label.

    Captures both false-path (o predicted where it shouldn't be) and
    missed-path (not-o predicted where o was expected).
    """
    dec = _as_grid_int(decoded_grid)
    lab = _as_grid_int(label_grid)
    return dec != lab


def normalized_entropy(attn_row: np.ndarray) -> float:
    p = np.asarray(attn_row, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 0.0
    p = np.clip(p, 0.0, None)
    denom = float(p.sum())
    if denom <= 0.0:
        return 0.0
    p = p / denom
    nz = p > 0.0
    if not np.any(nz):
        return 0.0
    h = -np.sum(p[nz] * np.log(p[nz]))
    return float(h / np.log(max(p.size, 2)))


def violation_neighbor_count(query_idx: int, violation_mask: np.ndarray, kind: str = "4conn") -> int:
    """Count how many neighbors (under `kind`) are in the violation set."""
    q = _validate_query_idx(query_idx)
    viol_flat = _as_grid_bool(violation_mask).reshape(-1)
    neigh = neighborhood_indices(q, kind=kind)
    return int(np.sum(viol_flat[neigh]))


def head_metrics_for_maze_cell(
    attn_l_row: np.ndarray,
    attn_h_row: np.ndarray,
    query_idx: int,
    violation_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute L/H entropy and mass metrics under all three neighborhood defs.

    Violation mass is restricted to 4-connected neighborhood intersected with V
    (matches the current Sudoku paper definition: mass_viol = sum over N(q) ∩ V).
    """
    q = _validate_query_idx(query_idx)
    p_l = _safe_prob_row(attn_l_row)
    p_h = _safe_prob_row(attn_h_row)

    viol_flat = _as_grid_bool(violation_mask).reshape(-1)

    out: Dict[str, float] = {
        "entropy_L": float(normalized_entropy(p_l)),
        "entropy_H": float(normalized_entropy(p_h)),
    }
    out["entropy_delta_H_minus_L"] = out["entropy_H"] - out["entropy_L"]

    # Three neighborhood definitions
    for kind, tag in (("4conn", "nbr4"), ("8conn", "nbr8"), ("window5", "window5")):
        neigh = neighborhood_indices(q, kind=kind)
        mass_l = float(np.sum(p_l[neigh]))
        mass_h = float(np.sum(p_h[neigh]))
        out[f"{tag}_mass_L"] = mass_l
        out[f"{tag}_mass_H"] = mass_h
        out[f"{tag}_delta_L_minus_H"] = mass_l - mass_h

    # Violation mass restricted to N4 ∩ V (the primary definition)
    neigh4 = neighborhood_indices(q, kind="4conn")
    viol_and_neigh = neigh4[viol_flat[neigh4]]
    viol_l = float(np.sum(p_l[viol_and_neigh])) if viol_and_neigh.size > 0 else 0.0
    viol_h = float(np.sum(p_h[viol_and_neigh])) if viol_and_neigh.size > 0 else 0.0
    out["violation_mass_L"] = viol_l
    out["violation_mass_H"] = viol_h
    out["violation_delta_L_minus_H"] = viol_l - viol_h

    return out


def head_score_from_metrics(metrics: Mapping[str, float]) -> float:
    """Analog of Sudoku head score: (delta_nbr4) + (delta_viol) + 0.5 * (delta_ent)."""
    nbr = float(metrics.get("nbr4_delta_L_minus_H", 0.0))
    viol = float(metrics.get("violation_delta_L_minus_H", 0.0))
    ent = float(metrics.get("entropy_delta_H_minus_L", 0.0))
    return float(nbr + viol + 0.5 * ent)


__all__ = [
    "GRID_SIZE",
    "NUM_CELLS",
    "TOKEN_PAD",
    "TOKEN_WALL",
    "TOKEN_SPACE",
    "TOKEN_START",
    "TOKEN_GOAL",
    "TOKEN_PATH",
    "neighborhood_indices",
    "compute_maze_violation_mask",
    "normalized_entropy",
    "violation_neighbor_count",
    "head_metrics_for_maze_cell",
    "head_score_from_metrics",
]
