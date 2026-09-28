"""
greedy.py — Greedy Nearest-Neighbor schedule heuristic for film location ordering.

The greedy solver accepts the output of all-pairs Dijkstra (a 2-D cost matrix)
as its distance source, which cleanly separates shortest-path computation from
schedule selection.

Algorithm (Nearest Neighbor):
    1. Start at the given start node.
    2. At each step, move to the closest unvisited scene node.
    3. Repeat until all scene nodes have been visited.
    4. Return the order, total cost, and per-step cost breakdown.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from stripboard.geo.dijkstra import all_pairs_shortest_paths, INF
from stripboard.geo.graph import SpatialGraph


@dataclass
class GreedyResult:
    """
    Output of a greedy nearest-neighbor schedule run.

    Attributes:
        order         : Sequence of node indices in visit order (includes start).
        total_cost    : Sum of all transition costs along the schedule.
        cost_breakdown: Per-step costs; cost_breakdown[k] is the cost of moving
                        from order[k] to order[k+1].
    """
    order: List[int]
    total_cost: float
    cost_breakdown: List[float]

    def __str__(self) -> str:
        steps = " → ".join(str(v) for v in self.order)
        return (f"GreedyResult(order=[{steps}], "
                f"total_cost={self.total_cost:.4f}, "
                f"steps={len(self.cost_breakdown)})")


def greedy_nearest_neighbor(dist_matrix: List[List[float]],
                             start: int,
                             scene_indices: Optional[List[int]] = None
                             ) -> GreedyResult:
    """
    Greedy Nearest-Neighbor heuristic for tour/schedule ordering.

    Given a precomputed all-pairs shortest-path cost matrix, builds a visit
    order by always choosing the cheapest unvisited node next.

    Args:
        dist_matrix  : n × n matrix where dist_matrix[i][j] is the shortest-path
                       cost from node i to node j (output of all_pairs_shortest_paths).
        start        : Index of the starting (basecamp / first scene) node.
        scene_indices: Subset of node indices representing "scenes" to be filmed.
                       If None, all nodes are treated as scenes (full tour).

    Returns:
        GreedyResult with the visit order, total cost, and per-step costs.

    Time complexity: O(k²) where k = number of scenes.
    """
    n = len(dist_matrix)

    # Determine which nodes must be visited
    if scene_indices is None:
        # All nodes except start; start is prepended automatically
        to_visit = set(range(n)) - {start}
    else:
        to_visit = set(scene_indices) - {start}

    order: List[int] = [start]
    cost_breakdown: List[float] = []
    total_cost = 0.0

    current = start

    while to_visit:
        best_cost = INF
        best_node = -1

        for candidate in to_visit:
            c = dist_matrix[current][candidate]
            if c < best_cost:
                best_cost = c
                best_node = candidate

        if best_node == -1 or best_cost == INF:
            # Remaining nodes are unreachable; stop early
            break

        order.append(best_node)
        cost_breakdown.append(best_cost)
        total_cost += best_cost
        to_visit.remove(best_node)
        current = best_node

    return GreedyResult(order=order, total_cost=total_cost,
                        cost_breakdown=cost_breakdown)


def greedy_schedule(graph: SpatialGraph,
                    scene_indices: List[int],
                    start: int) -> GreedyResult:
    """
    High-level entry point: compute all-pairs Dijkstra on *graph*, then apply
    the greedy nearest-neighbor heuristic to order the given *scene_indices*.

    Args:
        graph        : The SpatialGraph containing all locations.
        scene_indices: List of node indices that correspond to filming scenes.
        start        : Index of the node where filming begins (basecamp or first scene).

    Returns:
        GreedyResult with the greedy visit order and associated costs.

    Notes:
        - The *start* node need not be in *scene_indices*; it serves only as the
          departure point.
        - All-pairs shortest paths are computed to correctly handle disconnected
          sub-paths in the original graph (e.g., to reach a forest node you may
          need to pass through intermediate urban nodes).
    """
    cost_matrix = all_pairs_shortest_paths(graph)
    return greedy_nearest_neighbor(cost_matrix, start, scene_indices)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from stripboard.data.synthetic import generate_toy_graph, create_film_benchmark

    print("=== Greedy Nearest-Neighbor self-test ===\n")

    # ---- Test 1: toy 6-node graph ----
    g = generate_toy_graph(6, seed=7)
    print(g.summary())

    result = greedy_schedule(g, list(range(g.n)), start=0)
    print(f"\nGreedy schedule (all nodes): {result}")
    print(f"  Visit order : {result.order}")
    print(f"  Cost steps  : {[f'{c:.2f}' for c in result.cost_breakdown]}")
    print(f"  Total cost  : {result.total_cost:.2f}")

    # ---- Test 2: subset of scenes ----
    scenes = [1, 3, 5]
    result2 = greedy_schedule(g, scenes, start=0)
    print(f"\nGreedy schedule (scenes {scenes} from node 0): {result2}")

    # ---- Test 3: film benchmark ----
    print("\n--- Film benchmark (8 scenes) ---")
    fb = create_film_benchmark(8)
    res_fb = greedy_schedule(fb, list(range(fb.n)), start=0)
    print(res_fb)
    for i, (node_idx, cost) in enumerate(
            zip(res_fb.order[1:], res_fb.cost_breakdown)):
        print(f"  step {i+1}: → {fb.nodes[node_idx].name:20s}  cost {cost:.2f}")
