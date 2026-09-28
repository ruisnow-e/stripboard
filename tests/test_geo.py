"""Geographic layer: graph schema, Dijkstra, road network, seeded generators."""

import pytest

from stripboard.data.productions import PRODUCTIONS, build_production
from stripboard.data.synthetic import (create_film_benchmark,
                                       generate_sparse_graph,
                                       generate_toy_graph)
from stripboard.geo.dijkstra import (INF, all_pairs_shortest_paths, dijkstra,
                                     shortest_path)
from stripboard.geo.graph import Edge, Node, SpatialGraph, TerrainType
from stripboard.geo.road_network import (GeoNode, assemble_road_network,
                                         haversine_km)


def floyd_warshall(graph):
    """Independent all-pairs reference, O(n^3)."""
    n = graph.n
    d = [[0.0 if i == j else (graph.weight(i, j) or INF) for j in range(n)]
         for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


# ---- graph schema ---------------------------------------------------------

def test_edge_weight_formula():
    a = Node(0, "a", TerrainType.URBAN, 0)
    b = Node(1, "b", TerrainType.MOUNTAIN, 1500)
    e = Edge.from_nodes(a, b, 10.0)
    # terrain = (1.0 + 2.0) / 2, elevation factor = 1500 / 3000
    assert e.terrain_difficulty == 1.5
    assert e.weight == pytest.approx(10.0 * (1 + 0.3 * 0.5) * 1.5)


def test_elevation_factor_is_capped():
    assert Edge.compute_weight(1.0, 9000.0, 1.0) == pytest.approx(1.3)


def test_matrix_is_symmetric_and_rejects_negative_weights():
    g = generate_toy_graph(6, seed=42)
    m = g.to_matrix()
    assert all(m[i][j] == m[j][i] for i in range(g.n) for j in range(g.n))
    with pytest.raises(ValueError):
        g.set_edge(0, 1, -1.0)


# ---- Dijkstra ---------------------------------------------------------------

@pytest.mark.parametrize("graph", [
    generate_toy_graph(6, seed=42),
    generate_toy_graph(8, seed=7),
    create_film_benchmark(8),
    create_film_benchmark(12),
    generate_sparse_graph(40, edge_prob=0.08, seed=99),
])
def test_all_pairs_matches_floyd_warshall(graph):
    got = all_pairs_shortest_paths(graph)
    ref = floyd_warshall(graph)
    for i in range(graph.n):
        for j in range(graph.n):
            assert got[i][j] == pytest.approx(ref[i][j])


def test_connected_graph_settles_every_node():
    for n in (10, 50, 100):
        g = generate_sparse_graph(n, edge_prob=0.05, seed=n)
        assert dijkstra(g, 0).visited_count == n


def test_path_reconstruction_costs_what_it_claims():
    g = create_film_benchmark(8)
    path, cost = shortest_path(g, 0, 7)
    assert path == [0, 1, 7]
    assert cost == pytest.approx(24.27)
    assert sum(g.weight(a, b) for a, b in zip(path, path[1:])) == pytest.approx(cost)


def test_unreachable_target():
    nodes = [Node(i, f"n{i}", TerrainType.URBAN, 0) for i in range(3)]
    g = SpatialGraph.load_from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]], nodes)
    assert shortest_path(g, 0, 2) == ([], INF)
    assert not g.is_connected()


def test_eight_scene_benchmark_matrix():
    """Row 0 of the all-pairs matrix printed in REPORT section 6.1."""
    row = all_pairs_shortest_paths(create_film_benchmark(8))[0]
    assert [round(v, 2) for v in row] == [
        0.00, 14.55, 52.89, 28.61, 12.68, 42.51, 22.01, 24.27]


# ---- synthetic data -------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: create_film_benchmark(8),
    lambda: generate_toy_graph(6, seed=2),
    lambda: generate_sparse_graph(20, edge_prob=0.08, seed=99),
])
def test_generators_are_reproducible_and_connected(make):
    a, b = make(), make()
    assert a.to_matrix() == b.to_matrix()
    assert a.is_connected()


# ---- road network ---------------------------------------------------------

def test_haversine_la_to_nyc():
    la = GeoNode(0, "la", TerrainType.URBAN, 0, lat=34.052, lon=-118.244)
    ny = GeoNode(1, "ny", TerrainType.URBAN, 0, lat=40.755, lon=-73.985)
    assert haversine_km(la, ny) == pytest.approx(3940, rel=0.01)


def test_three_location_road_network():
    nodes = [
        GeoNode(0, "downtown_la", TerrainType.URBAN, 89, True,
                lat=34.052, lon=-118.244, city="Los Angeles"),
        GeoNode(1, "griffith_observatory", TerrainType.MOUNTAIN, 351,
                lat=34.118, lon=-118.300, city="Los Angeles"),
        GeoNode(2, "midtown_manhattan", TerrainType.URBAN, 10,
                lat=40.755, lon=-73.985, city="New York"),
    ]
    g = assemble_road_network(nodes)
    assert g.is_connected()
    # Griffith is in the same place as the base, so it gets a local road;
    # New York is reached only from the first-listed LA location.
    assert g.weight(0, 1) > 0 and g.weight(0, 2) > 0 and g.weight(1, 2) == 0


@pytest.mark.parametrize("key", list(PRODUCTIONS))
def test_productions_build_connected_networks(key):
    graph, nodes, prod = build_production(key)
    assert graph.n == len(prod.locations) == len(nodes)
    assert graph.is_connected()
    assert nodes[0].is_basecamp
