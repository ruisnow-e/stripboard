"""Greedy baseline, the solver's guarantees, and the headline results."""

import pytest

from stripboard.data.productions import PRODUCTIONS, build_production
from stripboard.data.synthetic import create_film_benchmark, generate_toy_graph
from stripboard.geo.dijkstra import all_pairs_shortest_paths
from stripboard.pipeline import main
from stripboard.scheduling.greedy import greedy_nearest_neighbor, greedy_schedule
from stripboard.scheduling.solver import solve
from stripboard.scheduling.subset_dp import optimal_schedule


# ---- greedy -----------------------------------------------------------------

def test_greedy_eight_scene_benchmark():
    res = greedy_schedule(create_film_benchmark(8), list(range(8)), start=0)
    assert res.order == [0, 4, 1, 6, 7, 3, 5, 2]
    assert round(res.total_cost, 2) == 123.45


def test_greedy_visits_every_scene_once():
    g = generate_toy_graph(8, seed=7)
    res = greedy_schedule(g, list(range(g.n)), start=0)
    assert sorted(res.order) == list(range(g.n))
    assert sum(res.cost_breakdown) == pytest.approx(res.total_cost)


def test_greedy_scene_subset():
    g = generate_toy_graph(6, seed=7)
    res = greedy_schedule(g, [1, 3, 5], start=0)
    assert res.order[0] == 0 and sorted(res.order[1:]) == [1, 3, 5]


@pytest.mark.parametrize("n, greedy, optimum", [
    (6, 332.81, 304.88),
    (8, 123.45, 123.45),
    (12, 179.47, 140.07),
])
def test_greedy_against_proven_optimum(n, greedy, optimum):
    """The 'both layers together' table in the README and REPORT 6.3."""
    cost = all_pairs_shortest_paths(create_film_benchmark(n))
    assert round(greedy_nearest_neighbor(cost, start=0).total_cost, 2) == greedy
    assert round(optimal_schedule(cost, start=0).total_cost, 2) == optimum


# ---- solver and the three productions ---------------------------------------

@pytest.mark.parametrize("key, cost, regions", [
    ("la_la_land", 210.68, 9),
    ("forrest_gump", 12124.09, 10),
    ("tenet", 34283.38, 11),
])
def test_production_schedules(key, cost, regions):
    graph, _, _ = build_production(key)
    matrix = all_pairs_shortest_paths(graph)
    report = solve(matrix, start=0)
    res = report.result
    assert report.mode == "scheduled" and report.regions == regions
    assert round(res.total_cost, 2) == cost
    # A valid schedule: every location once, legs summing to the total.
    assert sorted(res.order) == list(range(graph.n))
    assert sum(res.leg_costs) == pytest.approx(res.total_cost)
    walked = sum(matrix[a][b] for a, b in zip(res.order, res.order[1:]))
    assert walked == pytest.approx(res.total_cost)


def test_small_input_is_proven_optimal():
    cost = all_pairs_shortest_paths(create_film_benchmark(8))
    report = solve(cost, start=0)
    assert report.regions == 1
    assert report.guarantee == "globally optimal (proven)"


def test_closed_schedule_returns_to_base():
    graph, _, _ = build_production("tenet")
    res = solve(all_pairs_shortest_paths(graph), return_to_base=True).result
    assert res.order[0] == res.order[-1] == 0
    assert sorted(res.order[:-1]) == list(range(graph.n))


def test_cli(capsys):
    assert main([]) == 0
    listing = capsys.readouterr().out
    assert all(key in listing for key in PRODUCTIONS)
    assert main(["no_such_film"]) == 1
    assert main(["la_la_land"]) == 0
    assert "TOTAL TRANSITION COST" in capsys.readouterr().out
