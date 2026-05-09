"""Maze L/H attention plots (30x30 grid, with maze-geometry overlay)."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse PIL-title helper from Sudoku pipeline
# (helpers below)
from metrics import (
    GRID_SIZE,
    NUM_CELLS,
    TOKEN_WALL,
    TOKEN_SPACE,
    TOKEN_START,
    TOKEN_GOAL,
    TOKEN_PATH,
    neighborhood_indices,
)


# Colors / styling
HEATMAP_CMAP = "YlOrRd"
COLOR_QUERY = "#1f77b4"     # blue outline for query cell
COLOR_NBR4 = "#ff7f0e"      # orange outline for 4-connected neighbors
COLOR_START = "#2ca02c"     # green
COLOR_GOAL = "#d62728"      # red
COLOR_WALL = "#3a3a3a"      # dark grey for walls (drawn below heatmap)
COLOR_PATH_OUTLINE = "#5a7b8b"  # muted outline for ground-truth solution


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


def _draw_maze_overlay(
    ax,
    input_grid: np.ndarray,
    label_grid: Optional[np.ndarray] = None,
    view_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Overlay walls, start/goal markers, and GT-path tint.

    `view_bounds` = (r0, r1, c0, c1). When set, overlay artists whose cell
    falls outside this range are skipped so they do not leak out of a
    zoomed viewport.
    """
    grid = np.asarray(input_grid, dtype=np.int64).reshape(GRID_SIZE, GRID_SIZE)

    def _in_view(r: int, c: int) -> bool:
        if view_bounds is None:
            return True
        r0, r1, c0, c1 = view_bounds
        return (r0 <= r < r1) and (c0 <= c < c1)

    # Walls
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not _in_view(r, c):
                continue
            tok = int(grid[r, c])
            if tok == TOKEN_WALL:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1.0, 1.0,
                        facecolor=COLOR_WALL, alpha=0.35,
                        edgecolor="none", zorder=2,
                    )
                )

    # Start / Goal markers
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not _in_view(r, c):
                continue
            tok = int(grid[r, c])
            if tok == TOKEN_START:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1.0, 1.0,
                        fill=False, edgecolor=COLOR_START, linewidth=2.2, zorder=6,
                    )
                )
                ax.text(c, r, "S", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=COLOR_START, zorder=7)
            elif tok == TOKEN_GOAL:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1.0, 1.0,
                        fill=False, edgecolor=COLOR_GOAL, linewidth=2.2, zorder=6,
                    )
                )
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=COLOR_GOAL, zorder=7)

    # Ground-truth solution path tinting is disabled: the colored overlay
    # was competing with the attention heatmap. The unobstructed heatmap
    # clearer L/H separation visually.


def _draw_query_outline(ax, query_idx: int) -> None:
    r, c = divmod(int(query_idx), GRID_SIZE)
    ax.add_patch(
        Rectangle(
            (c - 0.5, r - 0.5), 1.0, 1.0,
            fill=False, edgecolor=COLOR_QUERY, linewidth=2.4, zorder=9,
        )
    )


def _draw_nbr4_outline(ax, query_idx: int) -> None:
    neigh = neighborhood_indices(int(query_idx), kind="4conn")
    for idx in neigh:
        r, c = divmod(int(idx), GRID_SIZE)
        ax.add_patch(
            Rectangle(
                (c - 0.5, r - 0.5), 1.0, 1.0,
                fill=False, edgecolor=COLOR_NBR4,
                linewidth=1.3, alpha=0.95, zorder=8,
            )
        )


def _draw_violation_x(
    ax,
    violation_mask: np.ndarray,
    view_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Draw red X markers on violation cells (decoded != label)."""
    viol = np.asarray(violation_mask, dtype=bool).reshape(GRID_SIZE, GRID_SIZE)
    rs, cs = np.where(viol)
    if rs.size == 0:
        return
    if view_bounds is not None:
        r0, r1, c0, c1 = view_bounds
        keep = (rs >= r0) & (rs < r1) & (cs >= c0) & (cs < c1)
        rs, cs = rs[keep], cs[keep]
        if rs.size == 0:
            return
    ax.scatter(cs, rs, marker="x", color="red", s=36, linewidths=1.1, alpha=0.90, zorder=7)


def _apply_heatmap_panel(
    ax,
    panel_map: np.ndarray,
    *,
    query_idx: int,
    input_grid: np.ndarray,
    label_grid: Optional[np.ndarray],
    violation_mask: np.ndarray,
    vmin: float,
    vmax: float,
    show_axis_labels: bool = True,
    view_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Any:
    im = ax.imshow(panel_map, cmap=HEATMAP_CMAP, interpolation="nearest", vmin=vmin, vmax=vmax, zorder=1)
    _draw_maze_overlay(ax, input_grid, label_grid=label_grid, view_bounds=view_bounds)
    _draw_violation_x(ax, violation_mask, view_bounds=view_bounds)
    _draw_nbr4_outline(ax, query_idx)
    _draw_query_outline(ax, query_idx)

    # Light ticks every 5 cells for orientation
    ax.set_xticks(np.arange(0, GRID_SIZE, 5))
    ax.set_yticks(np.arange(0, GRID_SIZE, 5))
    ax.set_xticklabels([str(i + 1) for i in range(0, GRID_SIZE, 5)])
    ax.set_yticklabels([str(i + 1) for i in range(0, GRID_SIZE, 5)])
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    if show_axis_labels:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    return im


def render_maze_core_comparison(
    output_png: str,
    *,
    l_map: np.ndarray,
    h_map: np.ndarray,
    query_idx: int,
    head_idx: int,
    layer_idx: int,
    act_step: int,
    input_grid: np.ndarray,
    label_grid: np.ndarray,
    decoded_h_grid: np.ndarray,
    zoom_radius: int = 5,
) -> None:
    """Render L vs H attention heatmaps - ZOOM ONLY - for one maze query cell.

    Layout: 1 row x 2 cols, each panel is a (2*zoom_radius+1) window centered
    on the query. This is where mass concentrates and L/H differ visibly.
    Start/Goal markers from the full maze do not appear unless they fall
    inside the zoom window.
    """
    from metrics import compute_maze_violation_mask

    _style_defaults()
    q = int(query_idx)
    qr, qc = divmod(q, GRID_SIZE)
    l_row = np.asarray(l_map, dtype=np.float64)[q].reshape(GRID_SIZE, GRID_SIZE)
    h_row = np.asarray(h_map, dtype=np.float64)[q].reshape(GRID_SIZE, GRID_SIZE)
    vmin = float(min(np.min(l_row), np.min(h_row)))
    vmax = float(max(np.max(l_row), np.max(h_row)))
    violation_mask = compute_maze_violation_mask(decoded_h_grid, label_grid)

    r0 = max(0, qr - zoom_radius)
    r1 = min(GRID_SIZE, qr + zoom_radius + 1)
    c0 = max(0, qc - zoom_radius)
    c1 = min(GRID_SIZE, qc + zoom_radius + 1)
    view_bounds = (r0, r1, c0, c1)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.5), constrained_layout=False)
    im = _apply_heatmap_panel(
        axes[0], l_row,
        query_idx=q, input_grid=input_grid, label_grid=label_grid,
        violation_mask=violation_mask, vmin=vmin, vmax=vmax,
        view_bounds=view_bounds,
    )
    axes[0].set_xlim(c0 - 0.5, c1 - 0.5)
    axes[0].set_ylim(r1 - 0.5, r0 - 0.5)
    # Show every other tick to avoid label crowding for 2-digit maze coords.
    xt = [i for i in range(c0, c1) if (i - c0) % 2 == 0]
    yt = [i for i in range(r0, r1) if (i - r0) % 2 == 0]
    axes[0].set_xticks(xt)
    axes[0].set_yticks(yt)
    axes[0].set_xticklabels([str(i + 1) for i in xt], fontsize=18)
    axes[0].set_yticklabels([str(i + 1) for i in yt], fontsize=18)
    axes[0].set_title("L-mode", fontsize=18, fontweight="bold", pad=6)

    _apply_heatmap_panel(
        axes[1], h_row,
        query_idx=q, input_grid=input_grid, label_grid=label_grid,
        violation_mask=violation_mask, vmin=vmin, vmax=vmax,
        view_bounds=view_bounds,
    )
    axes[1].set_xlim(c0 - 0.5, c1 - 0.5)
    axes[1].set_ylim(r1 - 0.5, r0 - 0.5)
    axes[1].set_xticks(xt)
    axes[1].set_yticks(yt)
    axes[1].set_xticklabels([str(i + 1) for i in xt], fontsize=18)
    axes[1].set_yticklabels([str(i + 1) for i in yt], fontsize=18)
    axes[1].set_title("H-mode", fontsize=18, fontweight="bold", pad=6)

    # Match Sudoku core_comparison subplots_adjust + colorbar settings exactly
    # so the rendered Maze panels have identical pixel size to Sudoku.
    fig.subplots_adjust(left=0.07, right=0.87, bottom=0.08, top=0.93, wspace=0.28)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.025)
    cbar.set_label("Attention weight", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    # Matplotlib legend matching the Sudoku core_comparison call. Anchor at
    # fig-x=0.40 (panel-pair midpoint after subplots_adjust right=0.87) so the
    # legend visually centers over the L/H panels. Save with bbox_inches="tight"
    # + pad_inches=0.05 so the canvas expands horizontally to fit the legend
    # without clipping the leftmost label.
    legend_handles = [
        Line2D([0], [0], color=COLOR_QUERY, lw=2.5, label="query cell"),
        Line2D([0], [0], color=COLOR_NBR4, lw=2.0, label="4-conn neighbor"),
        Line2D([0], [0], marker="x", color="red", linestyle="None", markersize=10,
               markeredgewidth=2.0, label="H-state error"),
    ]
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


def render_maze_temporal_grid(
    output_png: str,
    *,
    steps: Sequence[int],
    l_rows: Sequence[np.ndarray],
    h_rows: Sequence[np.ndarray],
    query_idx: int,
    head_idx: int,
    layer_idx: int,
    input_grid: np.ndarray,
    label_grid: np.ndarray,
    decoded_h_grids: Sequence[np.ndarray],
    zoom_radius: int = 5,
) -> None:
    """Render L and H attention panels across selected ACT steps - ZOOM ONLY."""
    from metrics import compute_maze_violation_mask

    _style_defaults()
    n = len(steps)
    if n == 0:
        raise ValueError("steps must be non-empty")

    q = int(query_idx)
    qr, qc = divmod(q, GRID_SIZE)
    l_panels = [
        np.asarray(l_rows[i], dtype=np.float64).reshape(GRID_SIZE, GRID_SIZE)
        for i in range(n)
    ]
    h_panels = [
        np.asarray(h_rows[i], dtype=np.float64).reshape(GRID_SIZE, GRID_SIZE)
        for i in range(n)
    ]

    vmin = float(min(np.min(m) for m in (l_panels + h_panels)))
    vmax = float(max(np.max(m) for m in (l_panels + h_panels)))

    r0 = max(0, qr - zoom_radius)
    r1 = min(GRID_SIZE, qr + zoom_radius + 1)
    c0 = max(0, qc - zoom_radius)
    c1 = min(GRID_SIZE, qc + zoom_radius + 1)
    view_bounds = (r0, r1, c0, c1)

    fig_w = max(6.0, 2.8 * n + 1.5)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 6.5), sharey=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)

    last_im = None
    for col, step in enumerate(steps):
        viol = compute_maze_violation_mask(decoded_h_grids[col], label_grid)
        for row, (mode, panel) in enumerate((("L", l_panels[col]), ("H", h_panels[col]))):
            ax = axes[row, col]
            last_im = _apply_heatmap_panel(
                ax, panel,
                query_idx=q, input_grid=input_grid, label_grid=label_grid,
                violation_mask=viol, vmin=vmin, vmax=vmax,
                show_axis_labels=False,
                view_bounds=view_bounds,
            )
            ax.set_xlim(c0 - 0.5, c1 - 0.5)
            ax.set_ylim(r1 - 0.5, r0 - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False)
            if row == 0:
                ax.set_title(f"Cycle {int(step):02d}", fontsize=28, fontweight="bold", pad=12)
            if col == 0:
                ax.set_ylabel(f"{mode}-mode", fontsize=28)

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


__all__ = [
    "render_maze_core_comparison",
    "render_maze_temporal_grid",
]
