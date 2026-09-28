"""
subset_dp.py — Held-Karp subset DP for exact shooting-order scheduling.

Consumes the all-pairs cost matrix from the geographic layer, not a raw
adjacency matrix: there 0 means "no direct road", here it means "already here",
and passing the wrong one gives silently wrong answers.

dp[] and parent[] are flat `array` buffers rather than nested lists, which is
what puts n = 20 within reach.  Time O(2^n n^2), space O(2^n n).
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Tuple

INF = float("inf")

# Above this many locations the 2^n table stops fitting in memory / patience.
# Kept as a module constant so benchmarks and callers agree on the ceiling.
EXACT_LIMIT = 20


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScheduleResult:
    """One scheduling run: the order, its cost, and how much work it took."""
    order: List[int]                    # includes the base again if closed
    total_cost: float
    leg_costs: List[float] = field(default_factory=list)
    # (subset, last-location) states relaxed.  Counts the subset DP only, so on
    # a partitioned run it misses the work inside path_cost_table.
    states_settled: int = 0
    n_locations: int = 0
    closed: bool = False


# ---------------------------------------------------------------------------
# Held-Karp exact DP
# ---------------------------------------------------------------------------

def optimal_schedule(cost: List[List[float]],
                     start: int = 0,
                     return_to_base: bool = False) -> ScheduleResult:
    """
    Globally optimal shooting order by Held-Karp subset DP.

    dp[S][v] is the cheapest way to have shot exactly the locations in S while
    standing at v.  S only grows, so the states form a DAG and one pass in
    increasing mask order settles them all.  See REPORT section 3.3.

    Args:
        cost          : n x n all-pairs cost matrix; INF where unreachable.
        start         : Base / first location.
        return_to_base: If True the crew returns to `start` after the last stop.

    Returns:
        ScheduleResult with the order, cost, per-leg costs and states settled.

    Raises:
        ValueError: If the matrix is empty or not square, `start` is out of
                    range, or n exceeds EXACT_LIMIT.
    """
    n = len(cost)
    if n == 0:
        raise ValueError("Cost matrix is empty.")
    for i, row in enumerate(cost):
        if len(row) != n:
            raise ValueError(
                f"Row {i} has length {len(row)}, expected {n} (square matrix).")
    if not (0 <= start < n):
        raise ValueError(f"start={start} out of range for {n} locations.")
    if n > EXACT_LIMIT:
        raise ValueError(
            f"n={n} exceeds the exact-DP ceiling of {EXACT_LIMIT}. "
            f"Use the clustered or heuristic solver at this scale.")

    if n == 1:
        return ScheduleResult(order=[start], total_cost=0.0, leg_costs=[],
                              states_settled=1, n_locations=1,
                              closed=return_to_base)

    size = 1 << n
    full = size - 1

    # Flat buffers: index (mask, v) as mask * n + v.
    dp = array("d", [INF]) * (size * n)
    parent = array("b", [-1]) * (size * n)

    dp[(1 << start) * n + start] = 0.0
    states_settled = 0

    for mask in range(size):
        # Every reachable state contains the start location.
        if not (mask >> start) & 1:
            continue
        base = mask * n
        for v in range(n):
            if not (mask >> v) & 1:
                continue
            d = dp[base + v]
            if d == INF:
                continue
            states_settled += 1

            row = cost[v]
            for u in range(n):
                if (mask >> u) & 1:
                    continue
                w = row[u]
                if w == INF:
                    continue  # u not reachable from v
                nd = d + w
                idx = ((mask | (1 << u)) * n) + u
                if nd < dp[idx]:
                    dp[idx] = nd
                    parent[idx] = v

    # ---- Choose the terminal state ----
    best_cost = INF
    best_last = -1
    tail_cost = 0.0

    if return_to_base:
        for v in range(n):
            if v == start:
                continue
            d = dp[full * n + v]
            back = cost[v][start]
            if d == INF or back == INF:
                continue
            if d + back < best_cost:
                best_cost = d + back
                best_last = v
                tail_cost = back
    else:
        for v in range(n):
            d = dp[full * n + v]
            if d < best_cost:
                best_cost, best_last = d, v

    if best_last < 0 or best_cost == INF:
        # Some location is unreachable — no complete schedule exists.
        return ScheduleResult(order=[], total_cost=INF, leg_costs=[],
                              states_settled=states_settled, n_locations=n,
                              closed=return_to_base)

    # ---- Reconstruct the order by walking parent[] backwards ----
    order: List[int] = []
    mask, v = full, best_last
    while v != -1:
        order.append(v)
        p = parent[mask * n + v]
        mask ^= (1 << v)
        v = p
    order.reverse()

    leg_costs = [cost[order[k]][order[k + 1]] for k in range(len(order) - 1)]
    if return_to_base:
        order.append(start)
        leg_costs.append(tail_cost)

    return ScheduleResult(
        order=order,
        total_cost=best_cost,
        leg_costs=leg_costs,
        states_settled=states_settled,
        n_locations=n,
        closed=return_to_base,
    )


# ---------------------------------------------------------------------------
# Sub-problem primitive used by the clustered (tier-2) solver
# ---------------------------------------------------------------------------

def path_cost_table(cost: List[List[float]],
                    members: List[int]) -> Tuple[List[List[float]],
                                                 List[List[List[int]]]]:
    """
    For one group of locations, solve every (entry, exit) routing question at once.

    Returns the cost of the cheapest route that starts at members[a], visits
    every location in `members` exactly once, and finishes at members[b] — for
    all pairs (a, b).  This is what lets the clustered solver stitch groups
    together without guessing where to enter and leave each group.

    Implemented as one Held-Karp run per entry point: m runs of O(2^m * m^2),
    so O(2^m * m^3) overall.  Kept small by capping group size.

    Args:
        cost   : Full n x n all-pairs cost matrix (global indices).
        members: Global indices belonging to this group.

    Returns:
        (pc, paths) where pc[a][b] is the cost described above (INF if no such
        route exists) and paths[a][b] is the corresponding sequence of *global*
        location indices.
    """
    m = len(members)
    if m > EXACT_LIMIT:
        # m Held-Karp runs, so m times more expensive than optimal_schedule.
        raise ValueError(
            f"group of {m} locations exceeds the exact-DP ceiling of "
            f"{EXACT_LIMIT}; lower region_cap.")
    if m == 1:
        return [[0.0]], [[[members[0]]]]

    sub = [[cost[p][q] for q in members] for p in members]
    size = 1 << m
    full = size - 1

    pc = [[INF] * m for _ in range(m)]
    paths: List[List[List[int]]] = [[[] for _ in range(m)] for _ in range(m)]

    for a in range(m):
        dp = array("d", [INF]) * (size * m)
        parent = array("b", [-1]) * (size * m)
        dp[(1 << a) * m + a] = 0.0

        for mask in range(size):
            if not (mask >> a) & 1:
                continue
            base = mask * m
            for v in range(m):
                if not (mask >> v) & 1:
                    continue
                d = dp[base + v]
                if d == INF:
                    continue
                row = sub[v]
                for u in range(m):
                    if (mask >> u) & 1:
                        continue
                    w = row[u]
                    if w == INF:
                        continue
                    nd = d + w
                    idx = ((mask | (1 << u)) * m) + u
                    if nd < dp[idx]:
                        dp[idx] = nd
                        parent[idx] = v

        for b in range(m):
            d = dp[full * m + b]
            if d == INF:
                continue
            pc[a][b] = d
            seq: List[int] = []
            mask, v = full, b
            while v != -1:
                seq.append(members[v])
                p = parent[mask * m + v]
                mask ^= (1 << v)
                v = p
            seq.reverse()
            paths[a][b] = seq

    return pc, paths


# ---------------------------------------------------------------------------
# Brute-force reference implementation (correctness oracle)
# ---------------------------------------------------------------------------

def brute_force_schedule(cost: List[List[float]],
                         start: int = 0,
                         return_to_base: bool = False) -> ScheduleResult:
    """Exhaustive-search oracle for optimal_schedule.  O(n!), test use only."""
    n = len(cost)
    others = [v for v in range(n) if v != start]

    best_cost = INF
    best_perm: Tuple[int, ...] = ()

    for perm in permutations(others):
        total = 0.0
        cur = start
        ok = True
        for nxt in perm:
            w = cost[cur][nxt]
            if w == INF:
                ok = False
                break
            total += w
            cur = nxt
        if not ok:
            continue
        if return_to_base:
            w = cost[cur][start]
            if w == INF:
                continue
            total += w
        if total < best_cost:
            best_cost = total
            best_perm = perm

    if best_cost == INF:
        return ScheduleResult(order=[], total_cost=INF, n_locations=n,
                              closed=return_to_base)

    order = [start] + list(best_perm)
    leg_costs = [cost[order[k]][order[k + 1]] for k in range(len(order) - 1)]
    if return_to_base:
        leg_costs.append(cost[order[-1]][start])
        order.append(start)

    return ScheduleResult(order=order, total_cost=best_cost,
                          leg_costs=leg_costs, n_locations=n,
                          closed=return_to_base)


# ---------------------------------------------------------------------------
# Demo — a four-location matrix small enough to check by hand.
# The 560-case cross-check against brute force is tests/test_subset_dp.py.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  subset_dp — Held-Karp exact scheduling, hand-checked example")
    print("=" * 70)

    # 0<->2 costs 8 via node 1, since there is no cheaper direct route.
    mock = [
        [0.0,  5.0,  8.0,  8.0],
        [5.0,  0.0,  3.0, 10.0],
        [8.0,  3.0,  0.0,  7.0],
        [8.0, 10.0,  7.0,  0.0],
    ]
    res = optimal_schedule(mock, start=0)
    print(f"  open  : order={res.order} cost={res.total_cost:.2f} "
          f"legs={[f'{c:.1f}' for c in res.leg_costs]}")
    res_c = optimal_schedule(mock, start=0, return_to_base=True)
    print(f"  closed: order={res_c.order} cost={res_c.total_cost:.2f}")
