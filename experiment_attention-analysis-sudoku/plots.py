#!/usr/bin/env python3
"""Sudoku attention rendering helpers.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional in some lightweight envs
    pd = None  # type: ignore[assignment]

HEATMAP_CMAP = "YlOrRd"
COLOR_L = "steelblue"
COLOR_H = "coral"
COLOR_DELTA = "seagreen"
TOPK_ATTENDED_LABELS = 3

# Keep Sudoku overlays close to existing visual_sudoku style.
RELATION_COLORS = {
    "row": "#ff7f0e",
    "col": "#17becf",
    "box": "#9467bd",
}


def _style_defaults() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.linewidth": 1.0,
        }
    )


def _ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(os.fspath(path)))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _validate_query_idx(query_idx: int) -> int:
    q = int(query_idx)
    if q < 0 or q >= 81:
        raise ValueError(f"query_idx must be in [0, 80], got {query_idx}")
    return q


def _to_9x9(arr: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    out = np.asarray(arr, dtype=dtype)
    if out.size != 81:
        raise ValueError(f"{name} must have 81 elements; got shape {out.shape}")
    return out.reshape(9, 9)


def _to_bool_9x9(arr: Any, *, name: str) -> np.ndarray:
    return _to_9x9(arr, name=name, dtype=bool)


def _extract_attention_row(attn_like: Any, query_idx: int, *, name: str) -> np.ndarray:
    """Accept [81], [9,9], or [81,81] and return one attention row [81]."""
    q = _validate_query_idx(query_idx)
    arr = np.asarray(attn_like, dtype=np.float64)
    if arr.size == 81 * 81:
        return arr.reshape(81, 81)[q].reshape(81)
    if arr.size == 81:
        return arr.reshape(81)
    raise ValueError(
        f"{name} must contain 81 (row/map) or 6561 (full map) elements; got shape {arr.shape}"
    )


def _safe_prob_row(attn_row: np.ndarray) -> np.ndarray:
    p = np.asarray(attn_row, dtype=np.float64).reshape(81)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, a_min=0.0, a_max=None)
    s = float(p.sum())
    if s <= 0:
        return np.zeros_like(p)
    return p / s


def _token_to_digit(token_id: int) -> int:
    t = int(token_id)
    if 2 <= t <= 10:
        return t - 1
    if 1 <= t <= 9:
        return t
    return 0


def _tokens_to_digits_grid(grid_tokens: Any) -> np.ndarray:
    grid = _to_9x9(grid_tokens, name="grid_tokens", dtype=np.int64)
    digits = np.vectorize(_token_to_digit)(grid)
    return digits.astype(np.int64)


def _violation_mask_from_decoded(decoded_grid_tokens: Any) -> np.ndarray:
    digits = _tokens_to_digits_grid(decoded_grid_tokens)
    viol = np.zeros((9, 9), dtype=bool)

    for r in range(9):
        vals = digits[r, :]
        for d in range(1, 10):
            idx = np.where(vals == d)[0]
            if idx.size > 1:
                viol[r, idx] = True

    for c in range(9):
        vals = digits[:, c]
        for d in range(1, 10):
            idx = np.where(vals == d)[0]
            if idx.size > 1:
                viol[idx, c] = True

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = digits[br : br + 3, bc : bc + 3]
            for d in range(1, 10):
                idx = np.argwhere(block == d)
                if idx.shape[0] > 1:
                    for rr, cc in idx:
                        viol[br + int(rr), bc + int(cc)] = True

    return viol


def _relation_masks(query_idx: int) -> Dict[str, np.ndarray]:
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


def _relation_component_masks(query_idx: int) -> Dict[str, np.ndarray]:
    rel = _relation_masks(query_idx)
    row = rel["row"]
    col = rel["col"]
    box = rel["box"]

    row_only = row & (~col) & (~box)
    col_only = col & (~row) & (~box)
    box_only = box & (~row) & (~col)
    neighborhood = row | col | box

    qr, qc = divmod(_validate_query_idx(query_idx), 9)
    query = np.zeros((9, 9), dtype=bool)
    query[qr, qc] = True
    outside = (~neighborhood) & (~query)

    return {
        "row_only": row_only,
        "col_only": col_only,
        "box_only": box_only,
        "neighborhood": neighborhood,
        "outside": outside,
    }


def _metric_box_text(metrics: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not metrics:
        return None
    lines = []
    for key, val in metrics.items():
        if isinstance(val, (float, int, np.floating, np.integer)):
            lines.append(f"{key}={float(val):.3f}")
        else:
            lines.append(f"{key}={val}")
    return "\n".join(lines) if lines else None


def _draw_metrics_box(ax: plt.Axes, metrics: Optional[Mapping[str, Any]]) -> None:
    text = _metric_box_text(metrics)
    if not text:
        return
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, boxstyle="round,pad=0.22"),
        zorder=10,
    )


def _apply_heatmap_panel(
    ax: plt.Axes,
    panel_map: np.ndarray,
    *,
    query_idx: int,
    violation_mask: np.ndarray,
    attn_row: Optional[np.ndarray],
    solution_grid_tokens: Optional[Any],
    show_topk_labels: bool,
    vmin: float,
    vmax: float,
    show_axis_labels: bool = True,
) -> Any:
    im = ax.imshow(panel_map, cmap=HEATMAP_CMAP, interpolation="nearest", vmin=vmin, vmax=vmax)
    add_sudoku_grid_lines(ax)
    draw_relation_outlines(ax, query_idx)
    draw_query_outline(ax, query_idx)
    if solution_grid_tokens is not None:
        draw_query_solution_digit(ax, query_idx, solution_grid_tokens)
    draw_violation_markers(ax, violation_mask)
    if show_topk_labels and attn_row is not None and solution_grid_tokens is not None:
        draw_topk_solution_labels(
            ax,
            attn_row,
            solution_grid_tokens,
            k=TOPK_ATTENDED_LABELS,
            exclude_idx=query_idx,
        )
    if show_axis_labels:
        ax.set_xlabel("Column", fontsize=20)
        ax.set_ylabel("Row", fontsize=20)
    return im


def add_sudoku_grid_lines(ax: plt.Axes) -> None:
    """Draw 3x3 Sudoku boundaries and axis tick labels on an imshow panel."""
    for pos in (2.5, 5.5):
        ax.axhline(pos, color="black", lw=1.0, alpha=0.35)
        ax.axvline(pos, color="black", lw=1.0, alpha=0.35)
    ax.set_xticks(range(9), [str(i + 1) for i in range(9)])
    ax.set_yticks(range(9), [str(i + 1) for i in range(9)])


def draw_query_outline(ax: plt.Axes, query_idx: int) -> None:
    """Outline the query cell with a high-contrast rectangle."""
    q = _validate_query_idx(query_idx)
    r, c = divmod(q, 9)
    ax.add_patch(
        Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, fill=False, ec=COLOR_L, lw=2.2, zorder=8)
    )


def draw_query_solution_digit(ax: plt.Axes, query_idx: int, solution_grid_tokens: Any) -> int:
    """Draw the solution digit for the query cell inside the outlined cell."""
    q = _validate_query_idx(query_idx)
    r, c = divmod(q, 9)
    digits = _tokens_to_digits_grid(solution_grid_tokens)
    digit = int(digits[r, c])
    label = str(digit) if digit > 0 else "?"
    ax.text(
        c,
        r,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor=COLOR_L,
            linewidth=1.8,
            alpha=0.90,
            boxstyle="round,pad=0.18",
        ),
        zorder=10,
    )
    return digit


def draw_relation_outlines(ax: plt.Axes, query_idx: int) -> None:
    """Outline row/column/box relation cells for the query index."""
    masks = _relation_masks(query_idx)
    lw_map = {"row": 1.0, "col": 1.0, "box": 1.2}

    for name, mask in masks.items():
        rr, cc = np.where(mask)
        for r, c in zip(rr, cc):
            ax.add_patch(
                Rectangle(
                    (float(c) - 0.5, float(r) - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    ec=RELATION_COLORS[name],
                    lw=lw_map[name],
                    alpha=0.75,
                    zorder=6,
                )
            )


def draw_violation_markers(ax: plt.Axes, violation_mask: Any) -> None:
    """Draw x-markers on cells that participate in Sudoku-rule conflicts."""
    mask = _to_bool_9x9(violation_mask, name="violation_mask")
    rr, cc = np.where(mask)
    if rr.size == 0:
        return
    ax.scatter(
        cc,
        rr,
        marker="x",
        s=36,
        c="red",
        linewidths=1.1,
        alpha=0.90,
        zorder=7,
    )


def draw_topk_solution_labels(
    ax: plt.Axes,
    attn_row: Any,
    solution_grid_tokens: Any,
    k: int = TOPK_ATTENDED_LABELS,
    exclude_idx: Optional[int] = None,
) -> np.ndarray:
    """Label top-k attended cells with solution digits decoded from token IDs."""
    row = np.asarray(attn_row, dtype=np.float64).reshape(81)
    if exclude_idx is not None:
        row = row.copy()
        row[int(_validate_query_idx(exclude_idx))] = -np.inf
    kk = max(1, min(int(k), 81))
    top_idx = np.argsort(row)[-kk:][::-1]
    digits = _tokens_to_digits_grid(solution_grid_tokens)

    for idx in top_idx:
        r, c = divmod(int(idx), 9)
        digit = int(digits[r, c])
        label = str(digit) if digit > 0 else "?"
        ax.text(
            c,
            r,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.88, boxstyle="circle,pad=0.18"),
            zorder=9,
        )
    return top_idx


def render_core_comparison(
    output_png: str,
    l_map: Any,
    h_map: Any,
    query_idx: int,
    head_idx: int,
    layer_idx: int,
    act_step: int,
    decoded_h_grid_tokens: Any,
    solution_grid_tokens: Any,
    metrics_l: Dict[str, Any],
    metrics_h: Dict[str, Any],
    title_pad: int = 24,
    show_topk_labels: bool = False,
    show_metrics_box: bool = False,
) -> None:
    """Render side-by-side L/H attention heatmaps for one query cell.

    L/H panels always share the same color scale.
    """
    _style_defaults()
    q = _validate_query_idx(query_idx)

    l_row = _extract_attention_row(l_map, q, name="l_map")
    h_row = _extract_attention_row(h_map, q, name="h_map")
    l_panel = l_row.reshape(9, 9)
    h_panel = h_row.reshape(9, 9)

    vmin = float(min(np.min(l_panel), np.min(h_panel)))
    vmax = float(max(np.max(l_panel), np.max(h_panel)))
    violation_mask = _violation_mask_from_decoded(decoded_h_grid_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.5), constrained_layout=False)

    im = _apply_heatmap_panel(
        axes[0],
        l_panel,
        query_idx=q,
        violation_mask=violation_mask,
        attn_row=l_row,
        solution_grid_tokens=solution_grid_tokens,
        show_topk_labels=bool(show_topk_labels),
        vmin=vmin,
        vmax=vmax,
    )
    if show_metrics_box:
        _draw_metrics_box(axes[0], metrics_l)
    axes[0].set_title("L-mode", fontsize=18, fontweight="bold", pad=max(4.0, float(title_pad) * 0.25))
    axes[0].tick_params(labelsize=18)

    _apply_heatmap_panel(
        axes[1],
        h_panel,
        query_idx=q,
        violation_mask=violation_mask,
        attn_row=h_row,
        solution_grid_tokens=solution_grid_tokens,
        show_topk_labels=bool(show_topk_labels),
        vmin=vmin,
        vmax=vmax,
    )
    if show_metrics_box:
        _draw_metrics_box(axes[1], metrics_h)
    axes[1].set_title("H-mode", fontsize=18, fontweight="bold", pad=max(4.0, float(title_pad) * 0.25))
    axes[1].tick_params(labelsize=18)

    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.08, top=0.93, wspace=0.28)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.025)
    cbar.set_label("Attention weight", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    legend_handles = [
        Line2D([0], [0], color=COLOR_L, lw=2.5, label="query cell"),
        Line2D([0], [0], color=RELATION_COLORS["row"], lw=2.0, label="same row"),
        Line2D([0], [0], color=RELATION_COLORS["col"], lw=2.0, label="same column"),
        Line2D([0], [0], color=RELATION_COLORS["box"], lw=2.0, label="same box"),
        Line2D([0], [0], marker="x", color="red", linestyle="None", markersize=10,
               markeredgewidth=2.0, label="H-state violation"),
    ]
    if show_topk_labels:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                color="white",
                linestyle="None",
                markersize=9,
                label=f"solution digit on top-{TOPK_ATTENDED_LABELS}",
            )
        )
    # Legend rendered in matplotlib at the same point-size unit as axis labels.
    # Anchored at the TOP of the figure rect with loc="upper center" so the
    # legend stays inside figsize bounds; this lets us save with the full
    # figure bbox (not tight) so Sudoku and Maze core_comparison images come
    # out at exactly the same pixel dimensions and panel sizes match.
    # Anchor at the panel-pair midpoint (fig-x ~0.40). The 5-item legend at
    # fontsize 18 is wider than the panel area, so we save with
    # bbox_inches="tight" - matplotlib expands the saved canvas horizontally
    # to fit the legend, and the legend stays visually centered above the
    # L/H panels (Sudoku and Maze use the same anchor and the same crop logic
    # so their boards still match).
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.40, 0.99),
        handlelength=0.7,
        handletextpad=0.3,
        columnspacing=1.2,
        borderpad=0.2,
    )

    _ensure_parent_dir(output_png)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_temporal_grid(
    output_png: str,
    steps: Sequence[int],
    l_rows: Sequence[Any],
    h_rows: Sequence[Any],
    query_idx: int,
    head_idx: int,
    layer_idx: int,
    decoded_h_grids_tokens: Sequence[Any],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Render temporal panels with L in top row and H in bottom row.

    By default, color scale is shared across all temporal panels. Supplying `vmin`
    and `vmax` allows forcing an externally shared scale.
    """
    _style_defaults()
    q = _validate_query_idx(query_idx)

    n = len(steps)
    if n == 0:
        raise ValueError("steps must be non-empty")
    if len(l_rows) != n or len(h_rows) != n or len(decoded_h_grids_tokens) != n:
        raise ValueError("steps, l_rows, h_rows, and decoded_h_grids_tokens must have equal length")

    l_panels = [
        _extract_attention_row(l_rows[i], q, name=f"l_rows[{i}]").reshape(9, 9)
        for i in range(n)
    ]
    h_panels = [
        _extract_attention_row(h_rows[i], q, name=f"h_rows[{i}]").reshape(9, 9)
        for i in range(n)
    ]

    if vmin is None:
        vmin = float(min(np.min(m) for m in (l_panels + h_panels)))
    if vmax is None:
        vmax = float(max(np.max(m) for m in (l_panels + h_panels)))

    fig_w = max(6.0, 2.8 * n + 1.5)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 6.5), constrained_layout=False)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)

    last_im = None
    for col, step in enumerate(steps):
        violation_mask = _violation_mask_from_decoded(decoded_h_grids_tokens[col])
        for row, (mode, panel) in enumerate((("L", l_panels[col]), ("H", h_panels[col]))):
            ax = axes[row, col]
            last_im = _apply_heatmap_panel(
                ax,
                panel,
                query_idx=q,
                violation_mask=violation_mask,
                attn_row=None,
                solution_grid_tokens=None,
                show_topk_labels=False,
                vmin=float(vmin),
                vmax=float(vmax),
                show_axis_labels=False,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
            if row == 0:
                ax.set_title(f"Cycle {int(step):02d}", fontsize=28, fontweight="bold", pad=12)
            if col == 0:
                ax.set_ylabel(f"{mode}-mode", fontsize=28)
            else:
                ax.set_ylabel("")

    fig.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.04, wspace=0.10, hspace=-0.45)
    if last_im is not None:
        # Use ax=axes.ravel() so the panel layout is unaffected (otherwise the
        # axes grow and the L/H gap collapses), and rely on matplotlib's
        # default vertical centering on the combined axes bbox (which spans
        # bottom..top symmetrically about the figure midline).
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.70, pad=0.02)
        cbar.set_label("Attention weight", fontsize=28)
        cbar.ax.tick_params(labelsize=22)

    _ensure_parent_dir(output_png)
    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def render_quant_summary(output_png: str, summary_df: Any) -> None:
    """Render class-wise delta bars.

    Expected columns:
      - cell_class
      - delta_neighborhood
      - delta_violation
      - delta_entropy
    """
    _style_defaults()

    if pd is None:
        raise ImportError("pandas is required for render_quant_summary")

    df = summary_df.copy() if isinstance(summary_df, pd.DataFrame) else pd.DataFrame(summary_df)
    required = ["cell_class", "delta_neighborhood", "delta_violation", "delta_entropy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"summary_df missing required columns: {missing}")

    plot_df = df[required].copy()
    plot_df["cell_class"] = plot_df["cell_class"].astype(str)
    for col in required[1:]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    class_order = list(dict.fromkeys(plot_df["cell_class"].tolist()))
    grouped = (
        plot_df.groupby("cell_class", sort=False)[required[1:]]
        .agg(["mean", "std", "count"])
        .reindex(class_order)
    )

    x = np.arange(len(class_order), dtype=np.float64)
    width = 0.24

    metrics = [
        ("delta_neighborhood", r"Neighborhood $\Delta$", COLOR_L, -width),
        ("delta_violation", r"Violation $\Delta$", COLOR_H, 0.0),
        ("delta_entropy", r"Entropy $\Delta$", COLOR_DELTA, width),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10.4, 4.2), constrained_layout=False)

    for metric, label, color, offset in metrics:
        means = grouped[(metric, "mean")].to_numpy(dtype=np.float64)
        stds = grouped[(metric, "std")].to_numpy(dtype=np.float64)
        counts = grouped[(metric, "count")].to_numpy(dtype=np.float64)

        errs = np.zeros_like(means)
        valid = counts > 1
        errs[valid] = 1.96 * stds[valid] / np.sqrt(counts[valid])

        ax.bar(
            x + offset,
            means,
            width=width,
            color=color,
            alpha=0.92,
            label=label,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=errs,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )

    ax.axhline(0.0, color="black", lw=1.0, alpha=0.6, zorder=2)
    ax.set_xticks(x, class_order, rotation=18, ha="right")
    ax.set_ylabel("Delta value")
    ax.set_title("Quantitative Summary by Cell Class", pad=10)
    ax.grid(axis="y", alpha=0.22, zorder=1)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.87, bottom=0.24)

    _ensure_parent_dir(output_png)
    fig.savefig(output_png, dpi=350, bbox_inches="tight")
    plt.close(fig)


def render_relation_decomposition(
    output_png: str,
    query_idx: int,
    attn_l_row: Any,
    attn_h_row: Any,
    violation_mask: Any,
) -> None:
    """Render relation component masses for one query (L vs H + delta line)."""
    _style_defaults()
    q = _validate_query_idx(query_idx)

    p_l = _safe_prob_row(_extract_attention_row(attn_l_row, q, name="attn_l_row")).reshape(9, 9)
    p_h = _safe_prob_row(_extract_attention_row(attn_h_row, q, name="attn_h_row")).reshape(9, 9)
    viol = _to_bool_9x9(violation_mask, name="violation_mask")
    comp = _relation_component_masks(q)

    categories = ["row_only", "col_only", "box_only", "neighborhood", "violation", "outside"]
    labels = ["row-only", "col-only", "box-only", "neighborhood", "violation", "outside"]

    l_vals = np.array(
        [
            float(p_l[comp["row_only"]].sum()),
            float(p_l[comp["col_only"]].sum()),
            float(p_l[comp["box_only"]].sum()),
            float(p_l[comp["neighborhood"]].sum()),
            float(p_l[viol].sum()),
            float(p_l[comp["outside"]].sum()),
        ],
        dtype=np.float64,
    )
    h_vals = np.array(
        [
            float(p_h[comp["row_only"]].sum()),
            float(p_h[comp["col_only"]].sum()),
            float(p_h[comp["box_only"]].sum()),
            float(p_h[comp["neighborhood"]].sum()),
            float(p_h[viol].sum()),
            float(p_h[comp["outside"]].sum()),
        ],
        dtype=np.float64,
    )
    deltas = h_vals - l_vals

    x = np.arange(len(categories), dtype=np.float64)
    width = 0.34

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.3), constrained_layout=False)
    ax.bar(x - width / 2.0, l_vals, width=width, color=COLOR_L, alpha=0.92, label="L", zorder=3)
    ax.bar(x + width / 2.0, h_vals, width=width, color=COLOR_H, alpha=0.92, label="H", zorder=3)

    ax2 = ax.twinx()
    ax2.plot(x, deltas, color=COLOR_DELTA, marker="o", linewidth=1.8, markersize=4.5, label=r"$\Delta$(H-L)", zorder=5)
    ax2.axhline(0.0, color=COLOR_DELTA, linewidth=0.9, alpha=0.45, zorder=2)

    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Attention mass")
    ax2.set_ylabel("Delta")
    ax.set_title("Relation Decomposition (L vs H)", pad=10)
    ax.grid(axis="y", alpha=0.22, zorder=1)

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper right")

    r, c = divmod(q, 9)
    fig.suptitle(f"query r{r+1}c{c+1}", y=0.98, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.24)

    _ensure_parent_dir(output_png)
    fig.savefig(output_png, dpi=350, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "add_sudoku_grid_lines",
    "draw_query_outline",
    "draw_query_solution_digit",
    "draw_relation_outlines",
    "draw_violation_markers",
    "draw_topk_solution_labels",
    "render_core_comparison",
    "render_temporal_grid",
    "render_quant_summary",
    "render_relation_decomposition",
]
