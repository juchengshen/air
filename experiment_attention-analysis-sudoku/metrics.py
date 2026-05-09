"""Per-cell + per-head L/H attention metrics for Sudoku."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from numpy.typing import NDArray


def _validate_query_idx(query_idx: int) -> int:
    q = int(query_idx)
    if q < 0 or q >= 81:
        raise ValueError(f"query_idx out of range [0, 80]: {query_idx}")
    return q


def _as_grid_int(values: np.ndarray) -> NDArray[np.int64]:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size != 81:
        raise ValueError(f"expected 81 values for a 9x9 grid, got {arr.size}")
    return arr.reshape(9, 9)


def _as_grid_bool(values: np.ndarray) -> NDArray[np.bool_]:
    arr = np.asarray(values, dtype=bool)
    if arr.size != 81:
        raise ValueError(f"expected 81 values for a 9x9 grid, got {arr.size}")
    return arr.reshape(9, 9)


def _safe_prob_row(attn_row: np.ndarray) -> NDArray[np.float64]:
    row = np.asarray(attn_row, dtype=np.float64).reshape(-1)
    if row.size != 81:
        raise ValueError(f"attention row must contain 81 values, got {row.size}")
    row = np.clip(row, 0.0, None)
    denom = float(row.sum())
    if denom <= 0.0:
        return np.full(81, 1.0 / 81.0, dtype=np.float64)
    return row / denom


def _coerce_to_digits(grid_like: np.ndarray) -> NDArray[np.int64]:
    """Convert a grid to display digits in [1..9], using 0 for non-digits."""
    grid = _as_grid_int(grid_like)

    # Prefer token decode when tokenized values are clearly present.
    if np.any(grid >= 10):
        digits = np.vectorize(token_to_display_digit, otypes=[np.int64])(grid)
        return digits.astype(np.int64, copy=False)

    return np.where((grid >= 1) & (grid <= 9), grid, 0).astype(np.int64, copy=False)


def token_to_display_digit(token_id: int) -> int:
    """Map model token id to Sudoku digit 1..9, else 0 for blank/invalid."""
    token = int(token_id)
    if 2 <= token <= 10:
        return token - 1
    return 0


def compute_violation_mask(decoded_grid_tokens: np.ndarray) -> NDArray[np.bool_]:
    """Mark cells that participate in row/col/box digit conflicts."""
    digits = np.vectorize(token_to_display_digit, otypes=[np.int64])(
        _as_grid_int(decoded_grid_tokens)
    )
    violation = np.zeros((9, 9), dtype=bool)

    for r in range(9):
        row = digits[r, :]
        for d in range(1, 10):
            idx = np.where(row == d)[0]
            if idx.size > 1:
                violation[r, idx] = True

    for c in range(9):
        col = digits[:, c]
        for d in range(1, 10):
            idx = np.where(col == d)[0]
            if idx.size > 1:
                violation[idx, c] = True

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = digits[br : br + 3, bc : bc + 3]
            for d in range(1, 10):
                idx = np.argwhere(block == d)
                if idx.shape[0] > 1:
                    for rr, cc in idx:
                        violation[br + rr, bc + cc] = True

    return violation


def neighborhood_indices(query_idx: int) -> NDArray[np.int64]:
    """Return sorted Sudoku neighborhood indices (row union col union box)."""
    q = _validate_query_idx(query_idx)
    r, c = divmod(q, 9)

    row = {r * 9 + j for j in range(9)}
    col = {i * 9 + c for i in range(9)}
    br = (r // 3) * 3
    bc = (c // 3) * 3
    box = {(br + i) * 9 + (bc + j) for i in range(3) for j in range(3)}

    return np.asarray(sorted(row | col | box), dtype=np.int64)


def relation_masks(query_idx: int) -> Dict[str, NDArray[np.bool_]]:
    """Return 9x9 row/col/box relation masks for one query cell (excluding self)."""
    q = _validate_query_idx(query_idx)
    r, c = divmod(q, 9)

    row_mask = np.zeros((9, 9), dtype=bool)
    col_mask = np.zeros((9, 9), dtype=bool)
    box_mask = np.zeros((9, 9), dtype=bool)

    row_mask[r, :] = True
    col_mask[:, c] = True

    br = (r // 3) * 3
    bc = (c // 3) * 3
    box_mask[br : br + 3, bc : bc + 3] = True

    row_mask[r, c] = False
    col_mask[r, c] = False
    box_mask[r, c] = False

    return {"row": row_mask, "col": col_mask, "box": box_mask}


def normalized_entropy(attn_row: np.ndarray) -> float:
    """Compute entropy normalized to [0, 1] for a nonnegative vector."""
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


def violation_neighbor_count(query_idx: int, violation_mask: np.ndarray) -> int:
    """Count violated neighbors for a query cell, excluding the query cell."""
    q = _validate_query_idx(query_idx)
    viol_flat = _as_grid_bool(violation_mask).reshape(-1)
    neigh = neighborhood_indices(q)
    neigh = neigh[neigh != q]
    return int(np.sum(viol_flat[neigh]))


def ambiguity_score(query_idx: int, decoded_h_digits: np.ndarray) -> float:
    """Ambiguity proxy: sum of distinct nonzero digits in row, col, and box."""
    q = _validate_query_idx(query_idx)
    digits = _coerce_to_digits(decoded_h_digits)
    r, c = divmod(q, 9)
    br = (r // 3) * 3
    bc = (c // 3) * 3

    units = (
        digits[r, :],
        digits[:, c],
        digits[br : br + 3, bc : bc + 3].reshape(-1),
    )

    score = 0.0
    for unit in units:
        vals = unit[(unit >= 1) & (unit <= 9)]
        score += float(np.unique(vals).size)
    return score


def mask_shift_score_for_cell(
    query_idx: int,
    decoded_l_tokens_prev: np.ndarray,
    decoded_l_tokens_cur: np.ndarray,
    blank_token: int = 1,
) -> float:
    """Return 1.0 if query cell flips blank/nonblank from prev to cur, else 0.0."""
    q = _validate_query_idx(query_idx)
    prev_flat = _as_grid_int(decoded_l_tokens_prev).reshape(-1)
    cur_flat = _as_grid_int(decoded_l_tokens_cur).reshape(-1)

    prev_blank = int(prev_flat[q]) == int(blank_token)
    cur_blank = int(cur_flat[q]) == int(blank_token)
    return float(prev_blank != cur_blank)


def head_metrics_for_cell(
    attn_l_row: np.ndarray,
    attn_h_row: np.ndarray,
    query_idx: int,
    violation_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute L/H entropy and mass metrics, plus L-H or H-L deltas."""
    q = _validate_query_idx(query_idx)
    p_l = _safe_prob_row(attn_l_row)
    p_h = _safe_prob_row(attn_h_row)

    neigh = neighborhood_indices(q)
    viol = _as_grid_bool(violation_mask).reshape(-1)

    ent_l = normalized_entropy(p_l)
    ent_h = normalized_entropy(p_h)
    neigh_l = float(np.sum(p_l[neigh]))
    neigh_h = float(np.sum(p_h[neigh]))
    viol_l = float(np.sum(p_l[viol]))
    viol_h = float(np.sum(p_h[viol]))

    return {
        "entropy_L": float(ent_l),
        "entropy_H": float(ent_h),
        "neighborhood_mass_L": float(neigh_l),
        "neighborhood_mass_H": float(neigh_h),
        "violation_mass_L": float(viol_l),
        "violation_mass_H": float(viol_h),
        "delta_neighborhood_mass": float(neigh_l - neigh_h),
        "delta_violation_mass": float(viol_l - viol_h),
        "delta_entropy": float(ent_h - ent_l),
    }


def head_score_from_metrics(metrics: Mapping[str, float]) -> float:
    """Compute head score: (nbr_L-H) + (viol_L-H) + 0.5 * (ent_H-L)."""
    nbr_delta = float(
        metrics.get(
            "delta_neighborhood_mass",
            float(metrics.get("neighborhood_mass_L", 0.0))
            - float(metrics.get("neighborhood_mass_H", 0.0)),
        )
    )
    viol_delta = float(
        metrics.get(
            "delta_violation_mass",
            float(metrics.get("violation_mass_L", 0.0))
            - float(metrics.get("violation_mass_H", 0.0)),
        )
    )
    ent_delta = float(
        metrics.get(
            "delta_entropy",
            float(metrics.get("entropy_H", 0.0)) - float(metrics.get("entropy_L", 0.0)),
        )
    )
    return float(nbr_delta + viol_delta + 0.5 * ent_delta)


def cell_score(vn_count: float, mask_shift: float, ambiguity: float) -> float:
    """Compute cell score: 2*vn + 1.5*mask_shift + 0.5*ambiguity."""
    return float(2.0 * float(vn_count) + 1.5 * float(mask_shift) + 0.5 * float(ambiguity))


def score(norm_cell: float, norm_head: float, norm_viol: float, norm_step: float) -> float:
    """Composite per-head score combining cell salience, head salience,
    violation alignment, and ACT-step penalty."""
    return float(
        1.5 * float(norm_cell)
        + 1.5 * float(norm_head)
        + 1.0 * float(norm_viol)
        - 0.5 * float(norm_step)
    )


__all__ = [
    "token_to_display_digit",
    "compute_violation_mask",
    "neighborhood_indices",
    "relation_masks",
    "normalized_entropy",
    "violation_neighbor_count",
    "ambiguity_score",
    "mask_shift_score_for_cell",
    "head_metrics_for_cell",
    "head_score_from_metrics",
    "cell_score",
    "score",
]
