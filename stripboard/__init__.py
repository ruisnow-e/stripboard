"""
stripboard — A dual-layer film production optimization framework.

    geo/          what does each move cost?   terrain-weighted graph, Dijkstra
    scheduling/   in what order to shoot?     greedy, subset DP, partitioned DP
    data/         synthetic benchmarks and three real productions
    pipeline.py   both layers end to end (`python -m stripboard <production>`)
"""
