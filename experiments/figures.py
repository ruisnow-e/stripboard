"""
figures.py — Every figure in docs/figures/, regenerated from the code.

Geographic layer:
  1. Runtime curve (log-log): Dijkstra all-pairs vs Greedy vs n.
  2. Adjacency matrix heatmaps for the synthetic film benchmarks.

Scheduling layer:
  3. Exact-solver cost: time and states settled against location count.
  4. Cost above the proven optimum, partitioned vs greedy.
  5. Cost against the region size cap, one panel per production.
  6. Both schedules for one production, drawn on its real coordinates.

Run from the repository root (needs matplotlib and numpy):
    python -m experiments.figures
"""

from __future__ import annotations

import os
import time
from math import cos, radians
from typing import List, Tuple

# --- matplotlib configuration (non-interactive backend for script use) -----
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from experiments.benchmark import (TimingResult, run_dijkstra_benchmark,
                                   run_greedy_benchmark)
from stripboard.data.productions import PRODUCTIONS, build_production, subset_upto
from stripboard.data.synthetic import create_film_benchmark
from stripboard.geo.dijkstra import all_pairs_shortest_paths
from stripboard.geo.road_network import GeoNode, assemble_road_network
from stripboard.scheduling.clustered_dp import PartitionError, clustered_schedule
from stripboard.scheduling.greedy import greedy_nearest_neighbor
from stripboard.scheduling.subset_dp import EXACT_LIMIT, optimal_schedule

# Colour-blind-safe palette.
GREEDY = "#2a78d6"
SCHEDULED = "#eb6834"
EXACT = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

_PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "docs", "figures")


def _ensure_plots_dir() -> str:
    """Create and return the plots output directory."""
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    return _PLOTS_DIR


def _save(fig: plt.Figure, filename: str) -> str:
    """Save *fig* as a PNG file in the plots directory and close it."""
    path = os.path.join(_ensure_plots_dir(), filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _style(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
    """Muted axes and gridlines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)


# ---------------------------------------------------------------------------
# 1. Runtime curve
# ---------------------------------------------------------------------------

def plot_runtime_curves(dijkstra_results: List[TimingResult],
                        greedy_results: List[TimingResult],
                        filename: str = "runtime_curves.png") -> str:
    """
    Plot log-log runtime curves for Dijkstra all-pairs and Greedy.

    Args:
        dijkstra_results: TimingResult list from run_dijkstra_benchmark.
        greedy_results  : TimingResult list from run_greedy_benchmark.
        filename        : Output PNG file name inside docs/figures/.

    Returns:
        The path the figure was written to.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    d_x = [r.n_nodes for r in dijkstra_results]
    d_y = [r.time_ms for r in dijkstra_results]
    ax.plot(d_x, d_y, "o-", color="steelblue", linewidth=2,
            markersize=6, label="Dijkstra (all-pairs)")

    g_x = [r.n_nodes for r in greedy_results]
    g_y = [r.time_ms for r in greedy_results]
    ax.plot(g_x, g_y, "s--", color="darkorange", linewidth=2,
            markersize=6, label="Greedy Nearest-Neighbor")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Nodes (n)", fontsize=12)
    ax.set_ylabel("Runtime (ms)", fontsize=12)
    ax.set_title("Algorithm Runtime: Dijkstra vs Greedy (log-log)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 2. Adjacency matrix heatmap
# ---------------------------------------------------------------------------

def plot_graph_heatmap(matrix: List[List[float]],
                       title: str = "Adjacency Matrix",
                       filename: str = "graph_heatmap.png") -> str:
    """
    Plot a heatmap of an adjacency matrix.

    Zero entries (no edge) are shown in white; positive weights use a sequential
    colour map so heavier edges appear darker.

    Args:
        matrix  : 2-D list of floats (square adjacency matrix).
        title   : Plot title.
        filename: Output PNG file name inside docs/figures/.

    Returns:
        The path the figure was written to.
    """
    n = len(matrix)
    data = np.array(matrix, dtype=float)
    masked = np.where(data == 0, np.nan, data)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.6), max(4, n * 0.6)))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")

    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                   norm=mcolors.LogNorm(
                       vmin=np.nanmin(masked[masked > 0]) if np.any(masked > 0) else 1,
                       vmax=np.nanmax(masked) if np.any(~np.isnan(masked)) else 1))

    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = matrix[i][j]
                txt = f"{val:.0f}" if val > 0 else ""
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=max(6, 10 - n // 3),
                        color="black" if val < np.nanmax(masked) * 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Edge Weight")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=max(6, 10 - n // 5))
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=max(6, 10 - n // 5))
    ax.set_xlabel("Node Index", fontsize=11)
    ax.set_ylabel("Node Index", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 3. Exact-solver cost against location count
# ---------------------------------------------------------------------------

def plot_exact_scaling(key: str = "la_la_land",
                       filename: str = "exact_scaling.png") -> str:
    """
    Time and states settled for the exact solver, on growing real subsets.

    Both axes are logarithmic, so the doubling per added location shows up as a
    straight line.

    Args:
        key     : Which production's locations to take the subsets from.
        filename: Output file name inside docs/figures/.

    Returns:
        The path the figure was written to.
    """
    _, nodes, _ = build_production(key)
    ns, secs, states = [], [], []
    for n in range(4, EXACT_LIMIT + 1):
        sub = [GeoNode(id=i, name=d.name, terrain_type=d.terrain_type,
                       elevation_m=d.elevation_m, is_basecamp=(i == 0),
                       lat=d.lat, lon=d.lon, city=d.city)
               for i, d in enumerate(nodes[:n])]
        cost = all_pairs_shortest_paths(assemble_road_network(sub))
        t0 = time.perf_counter()
        res = optimal_schedule(cost, start=0)
        secs.append(time.perf_counter() - t0)
        states.append(res.states_settled)
        ns.append(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(ns, secs, color=EXACT, linewidth=2, marker="o", markersize=5,
             markeredgecolor="white", markeredgewidth=1.2)
    ax1.set_yscale("log")
    ax1.grid(True, axis="y", color=GRID, linewidth=0.8)
    _style(ax1, "locations", "seconds to prove the optimum")
    ax1.set_title(f"Time  ({secs[-1]:.1f} s at n = {ns[-1]})",
                  fontsize=11, color=INK, loc="left")

    ax2.plot(ns, states, color=EXACT, linewidth=2, marker="o", markersize=5,
             markeredgecolor="white", markeredgewidth=1.2)
    ax2.set_yscale("log")
    ax2.grid(True, axis="y", color=GRID, linewidth=0.8)
    _style(ax2, "locations", "DP states settled")
    ax2.set_title(f"States  ({states[-1]:,} at n = {ns[-1]})",
                  fontsize=11, color=INK, loc="left")

    fig.suptitle("Exact solver cost by location count (log scale)",
                 fontsize=13, color=INK, x=0.5, y=1.02)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 4. Cost above the proven optimum
# ---------------------------------------------------------------------------


def plot_optimality_gap(filename: str = "optimality_gap.png") -> str:
    """
    Percentage above the proven optimum, partitioned and greedy.

    Measured on the largest subset of each production the exact solver can still
    prove, so there is a true optimum to compare against.

    Args:
        filename: Output file name inside docs/figures/.

    Returns:
        The path the figure was written to.
    """
    titles, part_gaps, greedy_gaps = [], [], []
    for key in PRODUCTIONS:
        _, nodes, prod = build_production(key)
        sub = subset_upto(nodes, EXACT_LIMIT)
        cost = all_pairs_shortest_paths(assemble_road_network(sub))
        opt = optimal_schedule(cost, start=0).total_cost
        # always_partition, or a subset this size would skip the partition and
        # simply be the exact solver again.
        part = clustered_schedule(cost, start=0, always_partition=True)
        draft = greedy_nearest_neighbor(cost, start=0)
        gap = lambda v: 0.0 if abs(v - opt) < 1e-9 else (v - opt) / opt * 100
        titles.append(prod.title)
        part_gaps.append(gap(part.total_cost))
        greedy_gaps.append(gap(draft.total_cost))

    y = list(range(len(titles)))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    # Dots rather than bars: the partitioned gap is 0.00% on every production,
    # and a zero-width bar renders as nothing at all.
    for i, (pg, gg) in enumerate(zip(part_gaps, greedy_gaps)):
        ax.plot([pg, gg], [i, i], color=GRID, linewidth=2, zorder=1)
        ax.scatter([gg], [i], s=110, color=GREEDY, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.scatter([pg], [i], s=110, color=SCHEDULED, zorder=4,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(f"+{gg:.2f}%", (gg, i), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK)
    ax.axvline(0, color=EXACT, linewidth=2, zorder=2)
    ax.annotate("proven optimum", (0, len(titles) - 0.42), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=EXACT)
    ax.set_yticks(y)
    ax.set_yticklabels(titles, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.15, max(greedy_gaps) * 1.3)
    ax.set_ylim(len(titles) - 0.3, -0.7)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    _style(ax, "cost above the proven optimum (%)")
    ax.scatter([], [], s=110, color=SCHEDULED, label="Partitioned")
    ax.scatter([], [], s=110, color=GREEDY, label="Greedy draft")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Cost above the proven optimum",
                 fontsize=12, color=INK, loc="left", pad=12)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 5. Cost against the region size cap
# ---------------------------------------------------------------------------

def plot_region_cap(filename: str = "region_cap.png") -> str:
    """
    Cost against the region size cap, as a percentage above each production's
    own best, with the caps that cannot be partitioned marked.

    Args:
        filename: Output file name inside docs/figures/.

    Returns:
        The path the figure was written to.
    """
    caps = list(range(2, 13))
    fig, axes = plt.subplots(1, len(PRODUCTIONS), figsize=(12, 3.6),
                             sharey=True)
    for ax, key in zip(axes, PRODUCTIONS):
        graph, _, prod = build_production(key)
        cost = all_pairs_shortest_paths(graph)
        xs, ys, failed = [], [], []
        for cap in caps:
            try:
                ys.append(clustered_schedule(cost, start=0,
                                             region_cap=cap).total_cost)
                xs.append(cap)
            except PartitionError:
                failed.append(cap)
        best = min(ys)
        pct = [(v - best) / best * 100 for v in ys]
        ax.plot(xs, pct, color=SCHEDULED, linewidth=2, marker="o",
                markersize=5, markeredgecolor="white", markeredgewidth=1.2)
        for cap in failed:
            ax.axvspan(cap - 0.5, cap + 0.5, color=GRID, alpha=0.7, lw=0)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        _style(ax, "region size cap")
        ax.set_title(prod.title, fontsize=10.5, color=INK, loc="left")
        if failed:
            ax.text(failed[0] - 0.3, ax.get_ylim()[1] * 0.9,
                    "cannot partition", fontsize=8, color=MUTED, rotation=90,
                    va="top")
    axes[0].set_ylabel("% above this production's best", fontsize=10, color=INK)
    fig.suptitle("Cost by region size cap", fontsize=13, color=INK,
                 x=0.5, y=1.04)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 6. Both schedules on the map
# ---------------------------------------------------------------------------

def _blocks(nodes: List[GeoNode],
            order: List[int]) -> List[Tuple[str, float, float]]:
    """
    Collapse a location-level order to the places it moves between.

    At map scale the locations inside one place sit within a pixel or two of
    each other, so the readable unit is the place.
    """
    out: List[Tuple[str, float, float]] = []
    for v in order:
        place = nodes[v].city
        if out and out[-1][0] == place:
            continue
        members = [nd for nd in nodes if nd.city == place]
        out.append((place,
                    sum(nd.lat for nd in members) / len(members),
                    sum(nd.lon for nd in members) / len(members)))
    return out


def plot_routes(key: str = "la_la_land",
                filename: str = "routes_la_la_land.png") -> str:
    """
    Draw the greedy draft and the scheduled route on their real coordinates.

    Args:
        key     : Which production to draw.
        filename: Output file name inside docs/figures/.

    Returns:
        The path the figure was written to.
    """
    graph, nodes, prod = build_production(key)
    cost = all_pairs_shortest_paths(graph)
    routes = [("Greedy draft", greedy_nearest_neighbor(cost, start=0), GREEDY),
              ("Scheduled", clustered_schedule(cost, start=0), SCHEDULED)]

    scale = cos(radians(sum(nd.lat for nd in nodes) / len(nodes)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (label, res, colour) in zip(axes, routes):
        blocks = _blocks(nodes, list(res.order))
        xs = [lon * scale for _, _, lon in blocks]
        ys = [lat for _, lat, _ in blocks]
        ax.plot(xs, ys, color=colour, linewidth=1.8, zorder=2, alpha=0.9)
        ax.scatter(xs, ys, s=45, color=colour, edgecolors="white",
                   linewidths=1.5, zorder=3)
        ax.scatter([xs[0]], [ys[0]], s=190, facecolors="none",
                   edgecolors=INK, linewidths=1.4, zorder=4)
        # Labels collide where a production clusters, so try offsets in turn
        # and keep the first that clears everything already placed.
        placed: List[Tuple[float, float, float, float]] = []
        w_per_char, h_lab = 0.0062 * (max(xs) - min(xs)), 0.030 * (max(ys) - min(ys))
        for i, (place, _, _) in enumerate(blocks):
            text = f"{i + 1}. {place}"
            w_lab = len(text) * w_per_char
            for dx, dy, ha in ((0.008, 0.012, "left"), (-0.008, 0.012, "right"),
                               (0.008, -0.022, "left"), (-0.008, -0.022, "right"),
                               (0.008, 0.036, "left"), (-0.008, -0.046, "right")):
                ox = xs[i] + dx * (max(xs) - min(xs)) * 3
                oy = ys[i] + dy * (max(ys) - min(ys)) * 3
                x0 = ox if ha == "left" else ox - w_lab
                box = (x0, oy - h_lab / 2, x0 + w_lab, oy + h_lab / 2)
                if not any(box[0] < q[2] and q[0] < box[2]
                           and box[1] < q[3] and q[1] < box[3] for q in placed):
                    break
            placed.append(box)
            ax.annotate(text, (ox, oy), fontsize=7.5, color=MUTED, ha=ha,
                        va="center")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{label} — cost {res.total_cost:,.0f}",
                     fontsize=11, color=INK, loc="left")
        ax.margins(0.12)

    fig.suptitle(f"{prod.title} — {graph.n} locations, "
                 f"numbered in shooting order (circle = base)",
                 fontsize=13, color=INK, x=0.5, y=1.02)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating figures into {_PLOTS_DIR} ...")

    sizes = [10, 50, 100, 500]
    print("  collecting Dijkstra and greedy runtimes ...")
    path = plot_runtime_curves(run_dijkstra_benchmark(sizes),
                               run_greedy_benchmark(sizes))
    print(f"  {'plot_runtime_curves':<22s} -> {path}")
    for n_sc in [6, 8, 12]:
        fb = create_film_benchmark(n_sc)
        path = plot_graph_heatmap(fb.to_matrix(),
                                  title=f"{n_sc}-Scene Film Benchmark Adjacency Matrix",
                                  filename=f"film_benchmark_{n_sc}_heatmap.png")
        print(f"  {'plot_graph_heatmap':<22s} -> {path}")

    for fn in (plot_exact_scaling, plot_optimality_gap,
               plot_region_cap, plot_routes):
        print(f"  {fn.__name__:<22s} -> {fn()}")
    print("Done.")
