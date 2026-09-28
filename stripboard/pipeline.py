"""
pipeline.py — End-to-end pipeline: a film's locations in, its shooting schedule out.

    real filming locations
          |
          v
    [ geographic layer ]         terrain-weighted graph
          |                      Dijkstra all-pairs
          v
      cost matrix                "what does each move really cost?"
          |
          v
    [ scheduling layer ]         MST regions, then exact DP within and across
          |
          v
    shooting order               "so shoot them in this order"

Usage (from the repository root):
    python -m stripboard                    # list the available productions
    python -m stripboard la_la_land         # schedule one
    python -m stripboard tenet --closed     # ... and return to base at wrap
"""

from __future__ import annotations

import sys
from typing import List, Optional

from stripboard.data.productions import PRODUCTIONS, build_production, true_groups
from stripboard.geo.dijkstra import all_pairs_shortest_paths
from stripboard.scheduling.greedy import greedy_nearest_neighbor
from stripboard.scheduling.solver import solve


def run_pipeline(key: str, return_to_base: bool = False) -> None:
    """
    Run both layers on one production and print the resulting schedule.

    Args:
        key           : A key from productions.PRODUCTIONS.
        return_to_base: If True the crew returns to base after the last location.
    """
    graph, nodes, prod = build_production(key)
    groups = true_groups(nodes)

    print("=" * 78)
    print(f"  SHOOTING SCHEDULE — {prod.title}")
    print(f"  {prod.scale}"
          f"{', returning to base' if return_to_base else ''}")
    print("=" * 78)

    # ---- Layer 1: locations -> terrain-weighted graph -> cost matrix --------
    roads = sum(1 for i in range(graph.n) for j in range(i + 1, graph.n)
                if graph.weight(i, j) > 0)
    cost = all_pairs_shortest_paths(graph)

    print("\n  [1] Geographic layer")
    print(f"      {graph.n} locations in {len(groups)} places, {roads} roads")
    print(f"      Dijkstra -> {graph.n}x{graph.n} cost matrix")
    print(f"      source: {prod.source}")

    # ---- Layer 2: cost matrix -> shooting order ----------------------------
    report = solve(cost, start=0, return_to_base=return_to_base)
    res = report.result

    print("\n  [2] Scheduling layer")
    print(f"      regions   : {report.regions}"
          f"{'  (nothing to partition — solved exactly)' if report.regions == 1 else ''}")
    print(f"      guarantee : {report.guarantee}")
    print(f"      solved in : {report.elapsed_ms:.1f} ms")

    # ---- Output ------------------------------------------------------------
    print("\n  SHOOTING ORDER")
    print(f"  {'#':>3s}  {'location':<30s} {'place':<20s} {'move':>9s}")
    print("  " + "-" * 68)
    prev_place = None
    for step, v in enumerate(res.order):
        place = nodes[v].city
        shown = place if place != prev_place else ""
        prev_place = place
        move = "— base —" if step == 0 else f"{res.leg_costs[step - 1]:,.0f}"
        print(f"  {step + 1:>3d}  {nodes[v].name:<30s} {shown:<20s} {move:>9s}")
    print("  " + "-" * 68)
    print(f"  {'':>3s}  {'TOTAL TRANSITION COST':<30s} {'':<20s} "
          f"{res.total_cost:>9,.0f}")

    moves = sum(1 for a, b in zip(res.order, res.order[1:])
                if nodes[a].city != nodes[b].city)
    # An open schedule crosses between places g-1 times at best; a closed one
    # has to come home, so it needs g.
    fewest = len(groups) if return_to_base else len(groups) - 1
    print(f"\n  Moved between places {moves} times "
          f"(fewest possible: {fewest}).")

    draft = greedy_nearest_neighbor(cost, start=0)
    if not return_to_base and draft.total_cost > res.total_cost:
        saved = draft.total_cost - res.total_cost
        print(f"  The greedy draft costs {draft.total_cost:,.0f} — "
              f"this schedule saves {saved:,.0f} "
              f"({saved / draft.total_cost * 100:.1f}%).")


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point; returns the process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print("Available productions:\n")
        for key, prod in PRODUCTIONS.items():
            print(f"  {key:<16s} {prod.title:<24s} "
                  f"{len(prod.locations):>3d} locations — {prod.scale}")
        print("\nUsage: python -m stripboard <key> [--closed]")
        return 0
    if args[0] not in PRODUCTIONS:
        print(f"Unknown production {args[0]!r}. "
              f"Choose from: {', '.join(PRODUCTIONS)}")
        return 1
    run_pipeline(args[0], return_to_base="--closed" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
