"""Subset DP against exhaustive search — the proof of optimality (560 cases)."""

import random

import pytest

from stripboard.scheduling.subset_dp import (EXACT_LIMIT, INF,
                                             brute_force_schedule,
                                             optimal_schedule, path_cost_table)


def _walk(cost, order):
    return sum(cost[order[k]][order[k + 1]] for k in range(len(order) - 1))


def test_hand_checked_mock():
    # 0<->2 costs 8 via node 1, since there is no cheaper direct route.
    mock = [
        [0.0,  5.0,  8.0,  8.0],
        [5.0,  0.0,  3.0, 10.0],
        [8.0,  3.0,  0.0,  7.0],
        [8.0, 10.0,  7.0,  0.0],
    ]
    res = optimal_schedule(mock, start=0)
    assert res.order == [0, 1, 2, 3]
    assert res.total_cost == 15.0
    assert res.leg_costs == [5.0, 3.0, 7.0]

    closed = optimal_schedule(mock, start=0, return_to_base=True)
    assert closed.order[0] == closed.order[-1] == 0
    assert closed.total_cost == 23.0


def test_matches_brute_force():
    rng = random.Random(5800)
    checks = 0
    for n in range(2, 9):
        for _ in range(40):
            m = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    m[i][j] = m[j][i] = round(rng.uniform(1, 50), 2)
            for closed in (False, True):
                a = optimal_schedule(m, start=0, return_to_base=closed)
                b = brute_force_schedule(m, start=0, return_to_base=closed)
                checks += 1
                assert a.total_cost == pytest.approx(b.total_cost, abs=1e-9), \
                    f"n={n} closed={closed}"
                # The reconstructed order must actually cost what we claim.
                assert _walk(m, a.order) == pytest.approx(a.total_cost, abs=1e-9)
    assert checks == 560


def test_path_cost_table_matches_fixed_endpoints():
    rng = random.Random(7)
    n = 6
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j] = m[j][i] = round(rng.uniform(1, 50), 2)
    members = [1, 3, 4, 5]
    pc, paths = path_cost_table(m, members)
    for a in range(len(members)):
        for b in range(len(members)):
            if a == b:
                assert pc[a][b] == INF
                continue
            seq = paths[a][b]
            assert seq[0] == members[a] and seq[-1] == members[b]
            assert sorted(seq) == sorted(members)
            assert _walk(m, seq) == pytest.approx(pc[a][b])


def test_unreachable_location_gives_no_schedule():
    m = [[0.0, 1.0, INF], [1.0, 0.0, INF], [INF, INF, 0.0]]
    res = optimal_schedule(m, start=0)
    assert res.order == [] and res.total_cost == INF


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        optimal_schedule([])
    with pytest.raises(ValueError):
        optimal_schedule([[0.0, 1.0]])
    with pytest.raises(ValueError):
        optimal_schedule([[0.0]], start=1)
    n = EXACT_LIMIT + 1
    with pytest.raises(ValueError):
        optimal_schedule([[0.0] * n for _ in range(n)])
