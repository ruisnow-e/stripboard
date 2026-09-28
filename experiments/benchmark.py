"""
benchmark.py — Simple runtime comparison between Dijkstra and Greedy.

Times both algorithms on the same generated graphs as size grows, and prints
the greedy schedule for the standard 8-scene film benchmark. All graphs come
from stripboard.data.synthetic's seeded generators, so every run is
reproducible.

Run from the repository root:
    python -m experiments.benchmark
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional

from stripboard.data.synthetic import create_film_benchmark, generate_sparse_graph
from stripboard.geo.dijkstra import all_pairs_shortest_paths, dijkstra, shortest_path
from stripboard.scheduling.greedy import greedy_nearest_neighbor


@dataclass
class TimingResult:
    """Wall-clock time (ms) for one algorithm run at one graph size."""
    algo_name: str
    n_nodes: int
    time_ms: float


# ---------------------------------------------------------------------------
# Timing benchmarks
# ---------------------------------------------------------------------------

def run_dijkstra_benchmark(sizes: Optional[List[int]] = None) -> List[TimingResult]:
    """Time all-pairs Dijkstra on sparse graphs of increasing size."""
    sizes = sizes or [10, 50, 100, 500]
    results = []
    for n in sizes:
        graph = generate_sparse_graph(n, edge_prob=0.05, seed=n)
        t0 = time.perf_counter()
        all_pairs_shortest_paths(graph)
        t1 = time.perf_counter()
        results.append(TimingResult("Dijkstra (all-pairs)", n, (t1 - t0) * 1000.0))
    return results


def run_greedy_benchmark(sizes: Optional[List[int]] = None) -> List[TimingResult]:
    """Time the greedy nearest-neighbor heuristic (cost matrix precomputed, not timed)."""
    sizes = sizes or [10, 50, 100, 500]
    results = []
    for n in sizes:
        graph = generate_sparse_graph(n, edge_prob=0.05, seed=n)
        cost_matrix = all_pairs_shortest_paths(graph)
        t0 = time.perf_counter()
        greedy_nearest_neighbor(cost_matrix, start=0)
        t1 = time.perf_counter()
        results.append(TimingResult("Greedy NearestNeighbor", n, (t1 - t0) * 1000.0))
    return results


def print_table(results: List[TimingResult], title: str) -> None:
    """Print a simple formatted timing table."""
    print(f"\n{title}")
    print(f"  {'Algorithm':<22s}  {'N Nodes':>7s}  {'Time (ms)':>10s}")
    print("  " + "-" * 43)
    for r in results:
        print(f"  {r.algo_name:<22s}  {r.n_nodes:>7d}  {r.time_ms:>10.2f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CS5800 Film Location Scheduling — Runtime Benchmarks")
    print("=" * 60)

    sizes = [10, 50, 100, 500]
    print_table(run_dijkstra_benchmark(sizes), "Dijkstra All-Pairs Shortest Paths")
    print_table(run_greedy_benchmark(sizes), "Greedy Nearest-Neighbor Schedule")

    # ---- Single-source stats on the 8-scene film benchmark ----
    fb = create_film_benchmark(8)
    print("\nDijkstra Single-Source Stats (8-scene benchmark, source=0)")
    t0 = time.perf_counter()
    r = dijkstra(fb, 0)
    t1 = time.perf_counter()
    path, cost = shortest_path(fb, 0, fb.n - 1)
    print(f"  Nodes visited : {r.visited_count}")
    print(f"  Heap pushes   : {r.heap_pushes}")
    print(f"  Time          : {(t1 - t0) * 1000:.3f} ms")
    print(f"  Path 0 -> {fb.n - 1}     : {path}, cost={cost:.2f}")

    # ---- Greedy schedule for the 8-scene film benchmark ----
    print("\n8-Scene Film Benchmark — Greedy Schedule")
    cost_matrix = all_pairs_shortest_paths(fb)
    result = greedy_nearest_neighbor(cost_matrix, start=0)
    print(f"  Step 0: {fb.nodes[result.order[0]].name:<16s} [START]")
    for i, node in enumerate(result.order[1:]):
        print(f"  Step {i + 1}: {fb.nodes[node].name:<16s} +{result.cost_breakdown[i]:.2f}")
    print(f"  TOTAL COST: {result.total_cost:.2f}")
    print()
