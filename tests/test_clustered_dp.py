"""The partitioned path against exhaustive search (400 + 200 cases)."""

import random
from itertools import permutations

import pytest

from stripboard.scheduling.clustered_dp import clustered_schedule, find_regions
from stripboard.scheduling.subset_dp import (INF, brute_force_schedule,
                                             optimal_schedule)


def _walk(cost, order):
    return sum(cost[order[k]][order[k + 1]] for k in range(len(order) - 1))


def _random_matrix(rng, n):
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j] = m[j][i] = round(rng.uniform(1, 60), 2)
    return m


def block_oracle(cost, regions, start, closed):
    """
    Cheapest schedule that finishes each region before leaving it.

    Enumerates every region order and every route within every region.
    Exponential and useless in practice, which is the point: it is the answer
    the region DP has to match.
    """
    home = next(i for i, g in enumerate(regions) if start in g)
    others = [i for i in range(len(regions)) if i != home]
    best = INF
    for region_order in permutations(others):
        for home_route in permutations([v for v in regions[home]
                                        if v != start]):
            def walk(idx, seq):
                nonlocal best
                if idx == len(region_order):
                    total = sum(cost[a][b] for a, b in zip(seq, seq[1:]))
                    if closed:
                        total += cost[seq[-1]][start]
                    best = min(best, total)
                    return
                for route in permutations(regions[region_order[idx]]):
                    walk(idx + 1, seq + list(route))
            walk(0, [start, *home_route])
    return best


def test_single_region_reproduces_exact_optimum():
    """With one region the method must be plain subset DP, order and all."""
    rng = random.Random(5800)
    checks = 0
    for n in range(2, 10):
        for _ in range(25):
            m = _random_matrix(rng, n)
            for closed in (False, True):
                # always_partition keeps the region machinery in play instead
                # of short-circuiting to the exact solver.
                got = clustered_schedule(m, start=0, region_cap=n,
                                         return_to_base=closed,
                                         always_partition=True)
                truth = brute_force_schedule(m, start=0, return_to_base=closed)
                exact = optimal_schedule(m, start=0, return_to_base=closed)
                checks += 1
                assert got.total_cost == pytest.approx(truth.total_cost, abs=1e-9)
                assert _walk(m, got.order) == pytest.approx(got.total_cost, abs=1e-9)
                assert list(got.order) == list(exact.order)
    assert checks == 400


def test_several_regions_match_block_oracle():
    """
    The part a single region cannot reach: best_join, the region-level
    transitions, and the multi-step chain reconstruction.
    """
    rng = random.Random(1234)
    checks = 0
    region_counts = set()
    for n in range(4, 9):
        for _ in range(20):
            m = _random_matrix(rng, n)
            for cap in (2, 3):
                got = clustered_schedule(m, start=0, region_cap=cap,
                                         always_partition=True)
                regions = got.regions
                region_counts.add(len(regions))
                if len(regions) < 2:
                    continue
                truth = block_oracle(m, regions, 0, False)
                checks += 1
                assert got.total_cost == pytest.approx(truth, abs=1e-9)
                assert _walk(m, got.order) == pytest.approx(got.total_cost, abs=1e-9)
                assert sorted(got.order) == list(range(n))
    assert checks == 200
    assert min(region_counts) >= 2 and max(region_counts) == 7


def test_finds_obvious_clusters():
    pts = [(0, 0), (1, 1), (2, 0), (0.5, 2),
           (100, 0), (101, 1), (100.5, 2),
           (50, 200), (51, 201)]
    n = len(pts)
    dist = [[((pts[i][0] - pts[j][0]) ** 2
              + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
             for j in range(n)] for i in range(n)]
    regions = sorted(sorted(r) for r in find_regions(dist, region_cap=5))
    assert regions == [[0, 1, 2, 3], [4, 5, 6], [7, 8]]
