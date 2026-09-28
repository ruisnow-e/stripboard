"""
Geographic layer — what does each move cost?

Models filming locations as a terrain-weighted graph and runs Dijkstra from
every node.  Its output, `all_pairs_shortest_paths(graph)`, is the n x n cost
matrix every scheduler reads.
"""
