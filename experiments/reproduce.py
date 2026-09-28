"""
reproduce.py — Regenerate every measured table in docs/REPORT.md.

The report makes a lot of quantitative claims. This script recomputes all of
them from the committed code and data and prints them in the same order the
report presents them, so any number in section 6 can be checked without
taking the report's word for it.  Correctness is checked separately by the
test suite (`pytest`).

Timings depend on the machine and will not match the report exactly; costs,
counts and percentages are deterministic and should match to the last digit.

Run from the repository root:
    python -m experiments.reproduce            # everything (~30 s)
    python -m experiments.reproduce --quick    # skip the exact-solver scaling sweep
"""

from __future__ import annotations

import sys
import time
from typing import List

from stripboard.data.productions import (PRODUCTIONS, build_production,
                                         subset_upto, true_groups)
from stripboard.data.synthetic import create_film_benchmark
from stripboard.geo.dijkstra import all_pairs_shortest_paths
from stripboard.geo.graph import SpatialGraph
from stripboard.geo.road_network import (GeoNode, assemble_road_network,
                                         haversine_km)
from stripboard.scheduling.clustered_dp import (DEFAULT_REGION_CAP,
                                                MAX_REGIONS, PartitionError,
                                                _mst_edges, clustered_schedule,
                                                find_regions)
from stripboard.scheduling.greedy import greedy_nearest_neighbor
from stripboard.scheduling.subset_dp import EXACT_LIMIT, optimal_schedule


def _gap(value: float, optimum: float) -> float:
    """Percentage above the optimum, with float noise clamped to zero."""
    d = value - optimum
    return 0.0 if abs(d) < 1e-9 else d / optimum * 100.0


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def plain_distance_graph(nodes: List[GeoNode],
                         ref: SpatialGraph) -> SpatialGraph:
    """The same road network, priced by raw kilometres instead of by terrain."""
    n = len(nodes)
    adj = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if ref.weight(i, j) > 0:
                adj[i][j] = adj[j][i] = round(
                    haversine_km(nodes[i], nodes[j]), 2)
    return SpatialGraph.load_from_matrix(adj, nodes)


def greedy_vs_optimum_on_benchmarks() -> None:
    """REPORT 6.3 — 'How far the greedy draft lands from the optimum'."""
    rule("Greedy draft against the proven optimum (synthetic benchmarks)")
    print(f"  {'Benchmark':<12s} {'Greedy':>9s} {'Optimum':>9s} {'Gap':>8s}")
    for n in (6, 8, 12):
        cost = all_pairs_shortest_paths(create_film_benchmark(n))
        draft = greedy_nearest_neighbor(cost, start=0).total_cost
        opt = optimal_schedule(cost, start=0).total_cost
        print(f"  {f'{n} scenes':<12s} {draft:>9.2f} {opt:>9.2f} "
              f"{_gap(draft, opt):>+7.2f}%")


def scheduling_the_productions() -> None:
    """REPORT 6.4 — 'Scheduling the three productions'."""
    rule("Scheduling the three productions")
    print(f"  {'Production':<22s} {'Greedy':>11s} {'Scheduled':>11s} "
          f"{'Saving':>8s} {'Time':>9s} {'Regions':>8s} {'Moves':>8s}")
    for key in PRODUCTIONS:
        graph, nodes, prod = build_production(key)
        cost = all_pairs_shortest_paths(graph)
        draft = greedy_nearest_neighbor(cost, start=0)
        t0 = time.perf_counter()
        sched = clustered_schedule(cost, start=0)
        ms = (time.perf_counter() - t0) * 1000.0
        moves = sum(1 for a, b in zip(sched.order, sched.order[1:])
                    if nodes[a].city != nodes[b].city)
        saving = (draft.total_cost - sched.total_cost) / draft.total_cost * 100
        print(f"  {prod.title:<22s} {draft.total_cost:>11,.2f} "
              f"{sched.total_cost:>11,.2f} {saving:>7.2f}% {ms:>8.1f}ms "
              f"{len(sched.regions):>8d} {moves:>4d}/{len(true_groups(nodes))-1}")


def what_the_partition_costs() -> None:
    """REPORT 6.5 — 'What the partition costs'."""
    rule("What the partition costs (largest provable subset of each production)")
    print(f"  {'Production':<22s} {'n':>3s} {'Optimum':>11s} {'Exact':>8s} "
          f"{'Partitioned':>12s} {'gap':>7s} {'Greedy':>11s} {'gap':>7s}")
    for key in PRODUCTIONS:
        _, nodes, prod = build_production(key)
        sub = subset_upto(nodes, EXACT_LIMIT)
        cost = all_pairs_shortest_paths(assemble_road_network(sub))
        t0 = time.perf_counter()
        exact = optimal_schedule(cost, start=0)
        secs = time.perf_counter() - t0
        # Force the partition: at this size the method would otherwise skip it
        # and be exact, and the question here is what partitioning costs.
        part = clustered_schedule(cost, start=0, always_partition=True)
        draft = greedy_nearest_neighbor(cost, start=0)
        o = exact.total_cost
        print(f"  {prod.title:<22s} {len(sub):>3d} {o:>11,.2f} {secs:>7.2f}s "
              f"{part.total_cost:>12,.2f} "
              f"{_gap(part.total_cost, o):>+6.2f}% "
              f"{draft.total_cost:>11,.2f} "
              f"{_gap(draft.total_cost, o):>+6.2f}%")


def what_the_partition_discovers() -> None:
    """REPORT 6.6 — 'What the partition discovers'."""
    rule("What the partition discovers (it never sees the place names)")
    for key in PRODUCTIONS:
        graph, nodes, prod = build_production(key)
        cost = all_pairs_shortest_paths(graph)
        regions = find_regions(cost)
        pure = sum(1 for r in regions if len({nodes[v].city for v in r}) == 1)
        print(f"  {prod.title:<22s} {len(regions):>2d} regions found, "
              f"{pure} match a single real place "
              f"({len(true_groups(nodes))} places in the data)")
        for r in sorted(regions, key=len, reverse=True):
            places = sorted({nodes[v].city for v in r})
            mark = " " if len(places) == 1 else "*"
            print(f"      {mark} {len(r):>2d}  {', '.join(places)}")


def why_both_layers() -> None:
    """REPORT 6.7 — 'Why the pipeline needs both layers'."""
    rule("Why the pipeline needs both layers")
    print(f"  {'Production':<22s} {'Pairs':>7s} {'No direct road':>15s} "
          f"{'Share':>7s} {'Detour cheaper':>15s}")
    for key in PRODUCTIONS:
        graph, _, prod = build_production(key)
        cost = all_pairs_shortest_paths(graph)
        n = graph.n
        pairs = n * (n - 1) // 2
        none = sum(1 for i in range(n) for j in range(i + 1, n)
                   if graph.weight(i, j) == 0)
        short = sum(1 for i in range(n) for j in range(i + 1, n)
                    if graph.weight(i, j) > 0
                    and cost[i][j] < graph.weight(i, j) - 1e-9)
        print(f"  {prod.title:<22s} {pairs:>7d} {none:>15d} "
              f"{none / pairs * 100:>6.1f}% {short:>15d}")


def terrain_contribution() -> None:
    """REPORT 6.8 — 'What the terrain weighting contributes'."""
    rule("What the terrain weighting contributes")
    print(f"  {'Production':<22s} {'Ignoring terrain':>17s} "
          f"{'Non-urban':>10s} {'Shortest':>10s} {'Longest':>11s} {'Spread':>10s}")
    for key in PRODUCTIONS:
        graph, nodes, prod = build_production(key)
        cost = all_pairs_shortest_paths(graph)
        plain = all_pairs_shortest_paths(plain_distance_graph(nodes, graph))
        score = lambda o: sum(cost[a][b] for a, b in zip(o, o[1:]))
        real = score(clustered_schedule(cost, start=0).order)
        naive = score(clustered_schedule(plain, start=0).order)
        km = [haversine_km(nodes[i], nodes[j])
              for i in range(graph.n) for j in range(i + 1, graph.n)
              if graph.weight(i, j) > 0]
        non_urban = sum(1 for nd in nodes if nd.terrain_type.value != "urban")
        print(f"  {prod.title:<22s} {_gap(naive, real):>+16.2f}% "
              f"{non_urban / len(nodes) * 100:>9.0f}% {min(km):>9.1f}km "
              f"{max(km):>10,.0f}km {max(km) / min(km):>9,.0f}x")


def region_cap_sweep() -> None:
    """REPORT 6.9 — 'Choosing the region size cap'."""
    rule("Choosing the region size cap")
    caps = list(range(2, 13))
    print("  " + " " * 22 + "".join(f"{c:>13d}" for c in caps))
    for key in PRODUCTIONS:
        graph, _, prod = build_production(key)
        cost = all_pairs_shortest_paths(graph)
        cells = []
        for cap in caps:
            try:
                s = clustered_schedule(cost, start=0, region_cap=cap)
                cells.append(f"{s.total_cost:>13,.2f}")
            except PartitionError:
                cells.append(f"{'over limit':>13s}")
        print(f"  {prod.title:<22s}" + "".join(cells))
    print(f"\n  default region cap = {DEFAULT_REGION_CAP}")


def mst_break() -> None:
    """REPORT 3.4 — the ratio gap the region count is read from."""
    rule("Where the MST's edge weights break")
    for key in PRODUCTIONS:
        graph, _, prod = build_production(key)
        cost = all_pairs_shortest_paths(graph)
        w = sorted((e[0] for e in _mst_edges(cost)), reverse=True)
        # The rule searches the whole admissible range, so print it: the top few
        # weights alone hide where the decisive break actually is.
        limit = min(MAX_REGIONS, graph.n)
        best_ratio, best_k = 0.0, 0
        for k in range(2, limit + 1):
            if k - 1 >= len(w):
                break
            r = w[k - 2] / w[k - 1] if w[k - 1] > 0 else float("inf")
            if r > best_ratio:
                best_ratio, best_k = r, k
        print(f"  {prod.title}")
        print(f"    {'k':>3s} {'cut here':>12s} {'next kept':>12s} {'ratio':>8s}")
        for k in range(2, limit + 1):
            if k - 1 >= len(w):
                break
            r = w[k - 2] / w[k - 1] if w[k - 1] > 0 else float("inf")
            mark = "  <- rule picks this" if k == best_k else ""
            print(f"    {k:>3d} {w[k - 2]:>12,.1f} {w[k - 1]:>12,.1f} "
                  f"{r:>7.2f}x{mark}")
        print(f"    final region count after the size cap: "
              f"{len(find_regions(cost))}")


def exact_scaling() -> None:
    """REPORT 6.2 — 'What the exact solver reaches'."""
    rule("What the exact solver reaches (timings are machine-dependent)")
    _, nodes, _ = build_production("la_la_land")
    print(f"  {'Locations':>9s} {'Time':>10s} {'States settled':>16s}")
    for n in range(14, EXACT_LIMIT + 1):
        sub = [GeoNode(id=i, name=d.name, terrain_type=d.terrain_type,
                       elevation_m=d.elevation_m, is_basecamp=(i == 0),
                       lat=d.lat, lon=d.lon, city=d.city)
               for i, d in enumerate(nodes[:n])]
        cost = all_pairs_shortest_paths(assemble_road_network(sub))
        t0 = time.perf_counter()
        r = optimal_schedule(cost, start=0)
        print(f"  {n:>9d} {time.perf_counter() - t0:>9.2f}s "
              f"{r.states_settled:>16,d}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    print("=" * 78)
    print("  Reproducing the measured tables in docs/REPORT.md")
    print("=" * 78)
    print("\n  Correctness checks are the test suite:  pytest")

    greedy_vs_optimum_on_benchmarks()
    scheduling_the_productions()
    what_the_partition_costs()
    what_the_partition_discovers()
    why_both_layers()
    terrain_contribution()
    region_cap_sweep()
    mst_break()
    if quick:
        print("\n  (skipping the exact-solver scaling sweep; drop --quick for it)")
    else:
        exact_scaling()
    print()
