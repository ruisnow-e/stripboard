"""
dijkstra.py — Dijkstra's shortest-path algorithm using Python's heapq.

Stale heap entries (from a node being relaxed more than once) are skipped
lazily on pop rather than using decrease-key, which is the standard way to
use a plain binary heap for Dijkstra.

all_pairs_shortest_paths returns a cost matrix (2-D list) — the single value
the scheduling layer consumes.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

from stripboard.geo.graph import SpatialGraph

INF = float("inf")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DijkstraResult:
    """
    Output of a single Dijkstra run from one source node.

    Attributes:
        dist         : dist[v] = shortest distance from source to v (INF if unreachable).
        prev         : prev[v] = predecessor of v on the shortest path (-1 if none).
        visited_count: Number of nodes settled (popped and finalized).
        heap_pushes  : Number of heappush calls performed (for benchmarking).
        source       : The source node index used for this run.
    """
    dist: List[float]
    prev: List[int]
    visited_count: int
    heap_pushes: int
    source: int


# ---------------------------------------------------------------------------
# Dijkstra implementation
# ---------------------------------------------------------------------------

def dijkstra(graph: SpatialGraph, source: int) -> DijkstraResult:
    """
    Run Dijkstra's algorithm from *source* on *graph* using a binary heap
    (heapq) as the priority queue.

    Args:
        graph : A SpatialGraph instance.
        source: Index of the starting node (0-based).

    Returns:
        DijkstraResult with dist[], prev[], visited_count, heap_pushes, source.

    Time complexity: O((V + E) log V).
    Space complexity: O(V + E).
    """
    n = graph.n
    dist = [INF] * n
    prev = [-1] * n
    dist[source] = 0.0
    visited_count = 0
    heap_pushes = 1

    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        if d > dist[u]:
            continue  # stale entry: a shorter path to u was already found

        visited_count += 1

        for v, w in graph.neighbors(u):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
                heap_pushes += 1

    return DijkstraResult(
        dist=dist,
        prev=prev,
        visited_count=visited_count,
        heap_pushes=heap_pushes,
        source=source,
    )


def shortest_path(graph: SpatialGraph, source: int,
                  target: int) -> Tuple[List[int], float]:
    """
    Compute and reconstruct the shortest path from *source* to *target*.

    Args:
        graph : A SpatialGraph instance.
        source: Starting node index.
        target: Goal node index.

    Returns:
        A tuple (path, cost) where:
            path : List of node indices from source to target (inclusive).
                   Empty list if target is unreachable.
            cost : Total path cost (INF if unreachable).
    """
    result = dijkstra(graph, source)
    cost = result.dist[target]

    if cost == INF:
        return [], INF

    path: List[int] = []
    cur = target
    while cur != -1:
        path.append(cur)
        cur = result.prev[cur]
    path.reverse()

    if path[0] != source:
        return [], INF

    return path, cost


def all_pairs_shortest_paths(graph: SpatialGraph) -> List[List[float]]:
    """
    Compute all-pairs shortest-path costs using Dijkstra from every source node.

    Returns a 2-D cost matrix where cost_matrix[i][j] is the shortest-path
    distance from node i to node j (INF if j is unreachable from i). This
    matrix is the primary input consumed by every scheduler.

    Time complexity: O(V * (V + E) log V).
    """
    return [dijkstra(graph, src).dist for src in range(graph.n)]


def all_pairs_results(graph: SpatialGraph) -> Dict[int, DijkstraResult]:
    """
    Run Dijkstra from every source and return a dict keyed by source index.

    Useful for benchmarking (provides visited_count and heap_pushes per run).
    """
    return {src: dijkstra(graph, src) for src in range(graph.n)}


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from stripboard.data.synthetic import generate_toy_graph

    print("=== Dijkstra self-test ===")
    g = generate_toy_graph(6, seed=42)
    print(g.summary())
    print()

    src, tgt = 0, 5
    path, cost = shortest_path(g, src, tgt)
    print(f"Shortest path {src} → {tgt}: {path}, cost={cost:.2f}")

    print("\nAll-pairs cost matrix:")
    matrix = all_pairs_shortest_paths(g)
    header = "     " + "  ".join(f"{j:6d}" for j in range(g.n))
    print(header)
    for i, row in enumerate(matrix):
        vals = "  ".join(f"{v:6.2f}" if v < INF else "   inf" for v in row)
        print(f"  {i}: {vals}")
