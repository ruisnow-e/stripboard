"""
Scheduling layer — in what order should the locations be shot?

Every scheduler here reads only the all-pairs cost matrix from the geographic
layer: `greedy` for a fast draft, `subset_dp` for a proven optimum up to 20
locations, `clustered_dp` for larger productions, and `solver` to pick between
them and report the guarantee that holds.
"""
