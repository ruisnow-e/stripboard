"""
clustered_dp.py — Partition the locations into regions, then solve exactly.

Solving a shooting order exactly costs 2^n and runs out around 20 locations.
Real productions get past that by block-shooting: finish one region, strike
camp, move on.  This module does the same thing:

    1. Group locations by building a minimum spanning tree (Kruskal with
       union-find) and deleting its heaviest edges — the long hauls.
    2. Solve each region exactly, for every entry and exit point rather than
       one (subset_dp.path_cost_table).
    3. Order the regions with a second DP whose state carries the exit
       location, so the joins are optimised with the ordering.

The partition is the only approximation; steps 2 and 3 are exact.  A production
that already fits the exact solver runs with a single region and returns a
proven optimum.  See REPORT sections 3.1 and 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from stripboard.scheduling.subset_dp import (EXACT_LIMIT, INF, ScheduleResult,
                                             optimal_schedule, path_cost_table)

# Cost is flat for caps 7-11 and worse at 12; below 6 Tenet needs more regions
# than MAX_REGIONS allows.  See REPORT section 6.9.
DEFAULT_REGION_CAP = 10
MAX_REGIONS = 13          # keeps the region-level 2^k DP cheap


class PartitionError(ValueError):
    """Raised when the locations cannot be split into solver-sized regions."""


# ---------------------------------------------------------------------------
# Region discovery (minimum spanning tree, cut at the widest gap)
# ---------------------------------------------------------------------------

def _mst_edges(cost: List[List[float]]) -> List[Tuple[float, int, int]]:
    """Kruskal MST over the cost matrix, as (weight, i, j) in accepted order."""
    n = len(cost)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    candidates = sorted((cost[i][j], i, j)
                        for i in range(n) for j in range(i + 1, n)
                        if cost[i][j] != INF)
    tree: List[Tuple[float, int, int]] = []
    for w, i, j in candidates:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            tree.append((w, i, j))
            if len(tree) == n - 1:
                break
    return tree


def _components(n: int, edges: List[Tuple[float, int, int]]) -> List[List[int]]:
    """Group locations by which ones remain connected through `edges`."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    groups: Dict[int, List[int]] = {}
    for v in range(n):
        groups.setdefault(find(v), []).append(v)
    return list(groups.values())


def find_regions(cost: List[List[float]],
                 region_cap: int = DEFAULT_REGION_CAP) -> List[List[int]]:
    """
    Split the locations into regions by cutting a minimum spanning tree.

    Delete the k-1 heaviest tree edges and k groups remain, each cheap to move
    around inside.  k is taken where the ratio between consecutive sorted
    weights is largest, so there is no threshold to tune; anything still larger
    than `region_cap` is then split again.  See REPORT section 3.4.

    Args:
        cost      : n x n all-pairs cost matrix.
        region_cap: Maximum locations per region.

    Returns:
        List of regions, each a list of location indices.

    Raises:
        PartitionError: If no admissible region count exists within MAX_REGIONS.
    """
    n = len(cost)
    if n <= region_cap:
        return [list(range(n))]

    tree = _mst_edges(cost)
    if len(tree) < n - 1:
        raise ValueError("Cost matrix is disconnected; cannot form regions.")
    heavy_first = sorted(tree, reverse=True)

    # --- Step 1: cut at the natural break in the tree's edge weights ---------
    cuts, best_ratio = 0, 0.0
    for k in range(2, min(MAX_REGIONS, n) + 1):
        if k - 1 >= len(heavy_first):
            break
        last_cut, first_kept = heavy_first[k - 2][0], heavy_first[k - 1][0]
        ratio = last_cut / first_kept if first_kept > 0 else INF
        if ratio > best_ratio:
            best_ratio, cuts = ratio, k - 1

    removed = set(range(cuts))          # indices into heavy_first

    # --- Step 2: split anything still too big for the exact solver ----------
    # Region by region rather than down the global weight order, so splitting
    # one oversized region leaves the others intact.
    while True:
        kept = [e for i, e in enumerate(heavy_first) if i not in removed]
        regions = _components(n, kept)
        oversized = [g for g in regions if len(g) > region_cap]
        if not oversized:
            break
        if len(regions) >= MAX_REGIONS:
            raise PartitionError(
                f"n={n} cannot be split into regions of at most {region_cap} "
                f"locations within the {MAX_REGIONS}-region limit.")
        target = set(max(oversized, key=len))
        inside = [i for i, (_, a, b) in enumerate(heavy_first)
                  if i not in removed and a in target and b in target]
        if not inside:
            raise PartitionError("Cannot split an oversized region any further.")
        removed.add(min(inside))        # heavy_first is sorted, so this is the
                                        # heaviest edge still inside the region
    return regions


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

@dataclass
class ClusteredResult(ScheduleResult):
    """A ScheduleResult that also records the regions it used."""
    regions: List[List[int]] = field(default_factory=list)


def clustered_schedule(cost: List[List[float]],
                       start: int = 0,
                       region_cap: int = DEFAULT_REGION_CAP,
                       return_to_base: bool = False,
                       always_partition: bool = False) -> ClusteredResult:
    """
    Partition the locations, then solve exactly inside each region and across
    them.

    A production that already fits the exact solver has nothing to partition and
    is routed straight to `optimal_schedule` — same answer as running the region
    machinery on a single region, but without solving every entry point when the
    entry is fixed.

    Args:
        cost            : n x n all-pairs cost matrix.
        start           : Base / first location.
        region_cap      : Maximum locations per region.
        return_to_base  : If True the crew returns to `start` at wrap.
        always_partition: Force the region machinery even when the input fits
                          the exact solver.  Used by the experiments and the
                          self-test; it can only make the answer worse.

    Returns:
        ClusteredResult with the order, cost and the regions that were used.
    """
    n = len(cost)
    if always_partition or n > EXACT_LIMIT:
        regions = find_regions(cost, region_cap=region_cap)
    else:
        regions = [list(range(n))]

    if len(regions) == 1 and not always_partition:
        exact = optimal_schedule(cost, start=start,
                                 return_to_base=return_to_base)
        return ClusteredResult(
            order=exact.order, total_cost=exact.total_cost,
            leg_costs=exact.leg_costs, states_settled=exact.states_settled,
            n_locations=n, closed=return_to_base,
            regions=regions)

    # The base must anchor whichever region contains it.
    home = next(i for i, g in enumerate(regions) if start in g)
    k = len(regions)

    # ---- Step 2: every entry/exit route inside each region, exactly ----
    tables = [path_cost_table(cost, g) for g in regions]

    # best_join[x][d][b]: cheapest way to stand at global location x, travel
    # into region d, cover it entirely, and come out at local index b.
    # Also remember which entry point achieved it, for reconstruction.
    best_join: Dict[Tuple[int, int, int], Tuple[float, int]] = {}
    for d in range(k):
        members = regions[d]
        pc, _ = tables[d]
        md = len(members)
        for x in range(n):
            row = cost[x]
            for b in range(md):
                bc, ba = INF, -1
                for a in range(md):
                    step = row[members[a]]
                    inner = pc[a][b]
                    if step == INF or inner == INF:
                        continue
                    tot = step + inner
                    if tot < bc:
                        bc, ba = tot, a
                if ba >= 0:
                    best_join[(x, d, b)] = (bc, ba)

    # ---- Step 3: region-level DP, state = (regions done, region, exit) ----
    # g[mask][(c, b)] = (cost, back-pointer)
    g: List[Dict[Tuple[int, int], Tuple[float, Optional[Tuple]]]] = [
        {} for _ in range(1 << k)]

    home_members = regions[home]
    pc_home, _ = tables[home]
    a_home = home_members.index(start)
    for b in range(len(home_members)):
        c = pc_home[a_home][b]
        if c < INF:
            g[1 << home][(home, b)] = (c, None)

    states = 0
    for mask in range(1 << k):
        if not (mask >> home) & 1 or not g[mask]:
            continue
        for (c, b), (dcost, _) in g[mask].items():
            states += 1
            x = regions[c][b]
            for d in range(k):
                if (mask >> d) & 1:
                    continue
                nm = mask | (1 << d)
                for b2 in range(len(regions[d])):
                    join = best_join.get((x, d, b2))
                    if join is None:
                        continue
                    nd = dcost + join[0]
                    cur = g[nm].get((d, b2))
                    if cur is None or nd < cur[0]:
                        g[nm][(d, b2)] = (nd, (mask, c, b))

    full = (1 << k) - 1
    if not g[full]:
        return ClusteredResult(order=[], total_cost=INF, n_locations=n,
                               closed=return_to_base, regions=regions)

    best_key, best_cost = None, INF
    for key, (c, _) in g[full].items():
        tail = cost[regions[key[0]][key[1]]][start] if return_to_base else 0.0
        if tail == INF:
            continue
        if c + tail < best_cost:
            best_cost, best_key = c + tail, key

    if best_key is None:
        return ClusteredResult(order=[], total_cost=INF, n_locations=n,
                               closed=return_to_base, regions=regions)

    # ---- Reconstruct: walk the region chain back, splicing in each route ----
    chain: List[Tuple[int, int, int]] = []   # (mask, region, exit local idx)
    mask, (c, b) = full, best_key
    while True:
        chain.append((mask, c, b))
        prev = g[mask][(c, b)][1]
        if prev is None:
            break
        mask, c, b = prev
    chain.reverse()

    order: List[int] = []
    for i, (_, c, b) in enumerate(chain):
        _, paths = tables[c]
        if i == 0:
            a = regions[c].index(start)
        else:
            prev_exit = regions[chain[i - 1][1]][chain[i - 1][2]]
            a = best_join[(prev_exit, c, b)][1]
        order.extend(paths[a][b])

    leg_costs = [cost[order[j]][order[j + 1]] for j in range(len(order) - 1)]
    if return_to_base:
        leg_costs.append(cost[order[-1]][start])
        order.append(start)

    return ClusteredResult(
        order=order,
        total_cost=best_cost,
        leg_costs=leg_costs,
        states_settled=states,
        n_locations=n,
        closed=return_to_base,
        regions=regions,
    )


# ---------------------------------------------------------------------------
# Demo — partitioning a map with obvious regions.
# The 400 + 200 case checks against exhaustive search are
# tests/test_clustered_dp.py.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("  clustered_dp — three well-separated clusters")
    print("=" * 72)
    pts = [(0, 0), (1, 1), (2, 0), (0.5, 2),
           (100, 0), (101, 1), (100.5, 2),
           (50, 200), (51, 201)]
    n = len(pts)
    dist = [[((pts[i][0] - pts[j][0]) ** 2
              + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
             for j in range(n)] for i in range(n)]
    for r in sorted(find_regions(dist, region_cap=5), key=len, reverse=True):
        print(f"      {r}")
    # always_partition, or n=9 would fit the exact solver and this would print
    # Held-Karp's answer while appearing to demonstrate the partitioned path.
    sched = clustered_schedule(dist, start=0, region_cap=5,
                               always_partition=True)
    exact = optimal_schedule(dist, start=0)
    print(f"      partitioned    order {sched.order}  cost {sched.total_cost:.2f}")
    print(f"      unpartitioned                    cost {exact.total_cost:.2f}"
          f"   ({(sched.total_cost - exact.total_cost) / exact.total_cost:+.2%}"
          f" for keeping the blocks together)")
