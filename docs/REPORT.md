# Stripboard: A Dual-Layer Film Production Optimization Framework

CS5800 Algorithms — final project report

---

## Contents

1. [Introduction](#1-introduction)
2. [The geographic layer](#2-the-geographic-layer)
3. [The scheduling layer](#3-the-scheduling-layer)
4. [Data](#4-data)
5. [Inputs, outputs and testing](#5-inputs-outputs-and-testing)
6. [Results](#6-results)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction

### Context

Film production is not just a creative process — it is a demanding logistical
operation. A crew relocating between filming locations must account for road
distance, terrain difficulty, elevation gain and equipment load. When shooting
schedules are determined informally, these transition costs accumulate into
significant budget overruns and schedule delays.

This project models film location scheduling as a **graph optimization
problem**, and splits it into two questions, each answered by one layer:

1. **What does each move cost?** Standard navigation tools compute driving
   distance but ignore terrain, elevation and equipment-hauling overhead. The
   geographic layer models locations as a terrain-weighted graph and runs
   Dijkstra from every node, producing an all-pairs cost matrix: the true
   cheapest cost of moving between any two locations, routed through
   intermediate stops where that is cheaper.
2. **In what order should the locations be shot?** Given that matrix, the
   scheduling layer chooses the order — a minimum-cost Hamiltonian path, which
   is NP-hard.

```
                filming locations
                       │
   ┌───────────────────▼────────────────────┐
   │  geographic layer                       │
   │  terrain-weighted graph                 │
   │  Dijkstra, all pairs                    │
   └───────────────────┬────────────────────┘
                       │   n × n cost matrix
   ┌───────────────────▼────────────────────┐
   │  scheduling layer                       │
   │  greedy draft · subset DP ·             │
   │  MST partitioning + region DP           │
   └───────────────────┬────────────────────┘
                       │
                shooting order
```

Neither half is usable alone. Across the three real productions studied here,
**81–85% of location pairs have no direct road between them** (Section 6.7), so
most entries in the matrix exist only because a shortest path was computed.

### Research question

> **Given a set of filming locations connected by terrain-weighted transit
> costs, what shooting order minimises the total cost of moving the crew
> between them — and how large a production can be scheduled with a provable
> guarantee rather than a heuristic?**

### Scope and the objective function

Locations are taken as already deduplicated: every scene sharing a location is
shot in one visit. Total cost splits as

```
total = Σ per-location shooting cost  +  Σ transition cost between locations
        └──── constant, order-independent ────┘   └──── the only term order affects ────┘
```

Every ordering pays the same shooting cost, so the optimal order is determined
entirely by transition cost.

The objective is therefore a minimum-cost Hamiltonian path from a fixed base
through all locations, the path variant of the Travelling Salesman Problem.
CLRS 3rd ed. §34.5.4 proves the decision version of the *tour* problem
NP-complete; the path variant reduces to it in the standard way, so the
minimisation problem here is NP-hard. Exact methods are therefore expected to
be exponential, and the interesting question is *how far* they reach.

---

## 2. The geographic layer

Code: `stripboard/geo/`

### 2.1 Graph schema

Each filming location is modeled as a **Node**:

| Field | Type | Description |
|---|---|---|
| `id` | int | 0-based index matching matrix row/column |
| `name` | str | Location name (e.g., `mountain_peak`) |
| `terrain_type` | enum | URBAN / FOREST / MOUNTAIN / COASTAL / DESERT |
| `elevation_m` | float | Elevation above sea level (metres) |
| `is_basecamp` | bool | True if this is the production start point |

Each transit route is modeled as an **Edge** with a composite weight formula:

```
w = distance × (1 + 0.3 × elevation_factor) × terrain_multiplier
```

Where:
- `elevation_factor = |Δelevation| / 3000` (normalized, capped at 1.0)
- `terrain_multiplier` = average of the two endpoint multipliers

**Terrain multipliers:**

| Terrain | Multiplier | Rationale |
|---|---|---|
| URBAN | 1.0 | Paved roads, easy crew access |
| COASTAL | 1.2 | Sand, limited vehicle access |
| FOREST | 1.4 | Dirt trails, tree clearance |
| DESERT | 1.6 | Sand dunes, heat stress, equipment risk |
| MOUNTAIN | 2.0 | Steep grades, altitude, unpaved |

The graph is stored as an adjacency matrix:
- `matrix[i][j] = 0` → no direct route between locations i and j
- `matrix[i][j] > 0` → edge weight (composite cost)

Note that this `0` means *no road*. In the all-pairs cost matrix the schedulers
read, `0` on the diagonal means *already here*; passing the adjacency matrix to
a scheduler by mistake gives silently wrong answers.

### 2.2 Dijkstra's shortest path

Dijkstra's algorithm computes the shortest path from one source node to all
other nodes in a weighted graph with non-negative edge weights. The
implementation uses Python's built-in `heapq` binary heap as the priority
queue, with lazy deletion for decrease-key: instead of updating an entry's
priority in place, a new `(distance, node)` pair is pushed whenever a shorter
path is found, and stale pairs are simply skipped when popped.

```
DIJKSTRA(graph, source):
    dist[v] ← ∞  for all v
    prev[v] ← -1 for all v
    dist[source] ← 0
    heap ← [(0, source)]          // heapq binary heap

    while heap is not empty:
        (d, u) ← heappop(heap)
        if d > dist[u]:          // lazy deletion: stale entry
            continue
        visited_count += 1

        for each neighbor v of u with edge weight w:
            if dist[u] + w < dist[v]:
                dist[v] ← dist[u] + w
                prev[v] ← u
                heappush(heap, (dist[v], v))

    return dist[], prev[]

PATH_RECONSTRUCTION(prev, source, target):
    path ← []
    cur ← target
    while cur ≠ -1:
        path.prepend(cur)
        cur ← prev[cur]
    return path

ALL_PAIRS_DIJKSTRA(graph):
    cost_matrix ← n×n matrix of ∞
    for each source in 0..n-1:
        result ← DIJKSTRA(graph, source)
        cost_matrix[source] ← result.dist
    return cost_matrix
```

**Time complexity:** O((V + E) log V) per source
**All-pairs complexity:** O(V · (V + E) log V)
**Space complexity:** O(V + E)

The all-pairs matrix is the single value that crosses into the scheduling
layer. Because it holds shortest paths, it satisfies the triangle inequality by
construction — a property Section 7 relies on.

### 2.3 Placing locations on the map

`Node` models a location by terrain and elevation but not by position.
`road_network.py` adds that: `GeoNode` extends `Node` with latitude, longitude
and the place (district, town or city) it belongs to, and distances are
great-circle kilometres (haversine). Edge weights still come from
`Edge.from_nodes`, unchanged.

`assemble_road_network` wires located nodes into two kinds of road, matching
how a crew moves:

- **Local roads.** Inside a place everything connects to everything, because a
  van can drive between any two sites in the same town.
- **Long haul.** Between places only the first-listed location connects, and
  only to its nearest few, so a cross-country move is a genuine routing
  problem. A spanning tree over those hubs is laid down first, because a pure
  nearest-neighbour long-haul network strands whole coasts.

---

## 3. The scheduling layer

Code: `stripboard/scheduling/`

### 3.1 The method

The scheduling layer is one method:

> **Partition the locations into regions, then solve exactly, inside each
> region and across them.**

When a production already fits the exact solver there is nothing to partition,
so the method runs with a single region and returns a proven global optimum.
Plain subset DP is this method's degenerate case, not a separate algorithm for
small inputs; Section 5 verifies that the two paths agree.

**The partition is the only approximation anywhere in the method.** Everything
downstream of it is exact: the route within each region, the order of the
regions, and where to leave one and enter the next.

The partition also matches practice. Productions block-shoot, finishing one
region before striking camp, so the constraint is one they already work under.

A greedy nearest-neighbour draft (Section 3.2) serves as the baseline every
result is measured against, and as the fallback when no partition can be
formed.

### 3.2 Greedy nearest-neighbour baseline

The greedy heuristic takes the Dijkstra cost matrix as input, not the raw
graph, which cleanly separates *how do we get there* from *in what order do we
go*.

```
GREEDY_NEAREST_NEIGHBOR(cost_matrix, start, scenes):
    visited ← {start}
    order   ← [start]
    total   ← 0
    current ← start

    while unvisited scenes remain:
        best_cost ← ∞
        best_node ← -1

        for each candidate in (scenes − visited):
            if cost_matrix[current][candidate] < best_cost:
                best_cost ← cost_matrix[current][candidate]
                best_node ← candidate

        order.append(best_node)
        total += best_cost
        visited.add(best_node)
        current ← best_node

    return order, total
```

**Time complexity:** O(k²) where k = number of scenes
**Space complexity:** O(k)

It gives a fast, valid schedule but no guarantee: choosing the nearest scene
now can strand the crew far from the remaining cluster later.

### 3.3 Exact ordering by subset DP

The state is a set and a position:

```
dp[S][v] = minimum cost to have shot exactly the locations in S,
           standing at location v, where v ∈ S
```

Because `S` only ever grows, the states form a directed acyclic graph, and the
recurrence is a shortest-path computation over it evaluated in order of
increasing `S`:

```
DP-SCHEDULE(cost, start):
    dp[{start}][start] ← 0,  all other dp ← ∞
    for each S in increasing order of bitmask:
        for each v ∈ S with dp[S][v] < ∞:
            for each u ∉ S:
                if dp[S][v] + cost[v][u] < dp[S ∪ {u}][u]:
                    dp[S ∪ {u}][u]     ← dp[S][v] + cost[v][u]
                    parent[S ∪ {u}][u] ← v
    return  min over v of dp[full][v]            (open schedule)
       or   min over v of dp[full][v] + cost[v][start]   (returning to base)
```

The order itself is recovered by walking `parent` backwards, clearing one bit
of the mask at each step (CLRS 3rd ed. §15.3, reconstructing an optimal
solution from the DP table).

This is the algorithm of Bellman (1962) and Held & Karp (1962). CLRS does not
contain it; the closest is Problem 15-3, a bitonic restriction of TSP. But
the paradigm it is built from is Chapter 15 (3rd ed.; Chapter 14 in the 4th):
optimal substructure, overlapping subproblems, and reconstruction from the
table.

**Time** `O(2ⁿ · n²)`  **Space** `O(2ⁿ · n)`

*Implementation note.* `dp` and `parent` are flat `array` buffers indexed as
`mask · n + v`, not nested Python lists. Nested lists of floats carry
per-object overhead that dominates at these sizes; the flat buffers hold raw
doubles and signed bytes. This is what puts n = 20 within reach on the test
machine rather than several sizes lower (Section 6.2).

### 3.4 Partitioning, and scheduling across regions

#### Finding the regions

Regions are found by cutting a minimum spanning tree:

```
FIND-REGIONS(cost, cap):
    T ← MST(cost)                          # Kruskal + union-find
    sort T's edges heaviest first
    cut the k-1 heaviest edges  →  k connected components
    while some region is larger than cap:
        cut the heaviest edge still inside the largest oversized region
    return the components
```

Deleting the `k-1` heaviest MST edges leaves `k` groups whose members are cheap
to reach from one another; a long haul between cities is the kind of expensive
edge this removes. Both ingredients are course material:
Kruskal's algorithm (CLRS 3rd ed. §23.2) and disjoint sets with path
compression (CLRS 3rd ed. Chapter 21). The clustering step itself is not a
separate algorithm: it is Kruskal stopped early. Its optimality (it maximises
the minimum spacing between groups) is the standard result in Kleinberg &
Tardos §4.7.

**How many regions?** The cut is taken at the largest ratio between consecutive
sorted weights, so there is no threshold to tune. The size cap then splits
anything still too big for the exact solver. The rule does most of the work at
every scale, and how cleanly it does so tracks how separated the geography
actually is:

| Production | Regions from the ratio rule | Largest ratio, at k | After the cap |
|---|---|---|---|
| La La Land | 7 | 1.75× at k = 7 | 9 |
| Forrest Gump | 8 | 2.81× at k = 8 | 10 |
| Tenet | 10 | **5.24×** at k = 10 | 11 |

*Tenet*'s blocks are separate countries and the break is unmistakable, a
5.24× drop. Inside a single city the same rule still finds 7 of the eventual 9
regions, but on a much shallower break: *La La Land*'s heaviest tree edges run
33.0, 20.5, 14.7, 14.1, and the best ratio is only 1.75×. In every case the cap
adds at most two more regions: two on the US productions, one on *Tenet*.

If a region is still too large for the exact solver, it is split again using
its own heaviest internal edge. Splitting region by region rather than
continuing down the global weight order matters: one outlying Tallinn location
can be peeled off its own block without shattering the cities around it.

**Time** `O(n² log n)`, dominated by Kruskal over the dense cost matrix.

#### Solving within a region — every way in and out

A region cannot be routed in isolation, because the cost of covering it depends
on where the crew enters and where it leaves. So for each region the exact
solver is run once per entry point, yielding the cheapest route that starts at
member `a`, covers the region, and finishes at member `b`, **for all pairs (a,
b)**:

```
PATH-COST-TABLE(cost, members):
    for each entry a:
        run DP-SCHEDULE restricted to members, starting at a
        for each exit b:  pc[a][b] ← dp[full][b];  record the path
    return pc, paths
```

**Time** `O(2^m · m³)` for a region of `m` locations.

#### Scheduling across regions

The region-level DP carries the exit location in its state, so the joins
between regions are optimised jointly with the region ordering rather than
chosen afterwards:

```
g[R][c][b] = minimum cost to have shot exactly the regions in R,
             standing at location b, the exit of region c

g[R ∪ {d}][d][y] = min over (c, b) ∈ g[R], over entries e of d:
                      g[R][c][b] + cost[b][e] + pc_d[e][y]
```

The inner minimisation over entry points `e` is precomputed once per (exit
location, target region, exit slot) triple, so the transition itself is `O(1)`.

**Time** `O(2^k · k² · m²)` for `k` regions of at most `m` locations.

Both exponents are now on small numbers, the region count and region size,
instead of on the location count.

| Step | Cost |
|---|---|
| Partitioning | `O(n² log n)` |
| Within a region of `m` locations | `O(2^m · m³)` |
| Across `k` regions | `O(2^k · k² · m²)` |
| Unpartitioned (subset DP alone) | `O(2ⁿ · n²)` time, `O(2ⁿ · n)` space |

---

## 4. Data

### 4.1 Synthetic benchmarks

`stripboard/data/synthetic.py` generates every synthetic graph —
`generate_toy_graph`, `generate_sparse_graph`, `create_film_benchmark` — with a
fixed random seed per size, so every number from them is reproducible directly
from the code; there are no separately hand-authored matrix files. Each is
guaranteed connected by laying down a random spanning tree before random edges
are added.

**8-scene film benchmark** (`create_film_benchmark(8)`). Raw adjacency matrix
(direct routes only; `0` = no direct edge). This is the input to Dijkstra,
before shortest-path costs are computed:

```
      0      1      2      3      4      5      6      7
0   0.00  14.55   0.00   0.00  12.68   0.00  35.80  29.81
1  14.55   0.00   0.00  14.06   0.00   0.00   7.46   9.72
2   0.00   0.00   0.00   0.00   0.00  10.38   0.00  36.54
3   0.00  14.06   0.00   0.00   0.00  24.79   0.00  23.73
4  12.68   0.00   0.00   0.00   0.00  37.63   0.00   0.00
5   0.00   0.00  10.38  24.79  37.63   0.00  20.50   0.00
6  35.80   7.46   0.00   0.00   0.00  20.50   0.00   0.00
7  29.81   9.72  36.54  23.73   0.00   0.00   0.00   0.00
```

| ID | Location | Terrain | Elevation (m) |
|---|---|---|---|
| 0 | downtown_plaza | urban | 4 |
| 1 | forest_trail | forest | 1,737 |
| 2 | mountain_peak | mountain | 1,347 |
| 3 | coastal_cove | coastal | 2,228 |
| 4 | desert_dunes | desert | 2,351 |
| 5 | ancient_ruins | mountain | 1,642 |
| 6 | river_crossing | forest | 1,872 |
| 7 | cliff_edge | coastal | 1,368 |

### 4.2 Three real productions

`stripboard/data/productions.py` holds 90 locations from three real
productions, chosen so that the same method can be tested at three geographic
scales:

| Production | Locations | Real places | Scale |
|---|---|---|---|
| La La Land (2016) | 26 | 12 Los Angeles districts | one city |
| Forrest Gump (1994) | 30 | 16 towns across 8 states | one country |
| Tenet (2020) | 34 | 11 cities across 7 countries | the world |

Coordinates are real latitude and longitude; distances are great-circle
kilometres; edge weights come from the geographic layer's terrain and elevation
formula, unchanged; the road network is assembled as in Section 2.3.

**Provenance.** Location lists are transcribed from published location guides
(`movie-locations.com`), which are enthusiast reconstructions, not production
paperwork. Each entry was then checked against OpenStreetMap and, where one
exists, an authoritative record; `productions.py` documents that pass and what
it changed. Coordinates are landmark-accurate for named landmarks and
street-accurate for entries given only as an address, an error of a few
hundred metres, immaterial next to the tens of kilometres between districts.
Real shooting order is not published for any of the three, so these schedules
can be compared with each other but not against what the crews actually did.

**Ground truth for the partition.** Each location carries the district, town or
city it truly belongs to. The scheduler never sees this field; it exists only
so the partition can be scored against it (Section 6.6).

---

## 5. Inputs, outputs and testing

### Geographic layer

**Inputs:**
- `graph` — a `SpatialGraph`; each node carries a terrain type and an
  elevation, and each edge carries the composite weight from Section 2.1
- `source` — index of the node to start from, normally the basecamp at index 0

**Outputs:**
- `dijkstra(graph, source)` returns a `DijkstraResult`: `dist` (cheapest cost
  to every node), `prev` (predecessors, for path reconstruction),
  `visited_count`, `heap_pushes`, and `source`
- `all_pairs_shortest_paths(graph)` returns the n × n cost matrix — the single
  value the scheduling layer consumes
- `shortest_path(graph, source, target)` returns `(path, cost)`, where `path`
  is the node sequence from source to target and is empty when the target is
  unreachable

### Scheduling layer

**Inputs.** Every scheduler takes the same arguments:

| Argument | Meaning |
|---|---|
| `cost` | an n × n matrix; `cost[i][j]` is the cheapest cost of moving the crew from location *i* to location *j*, produced by `all_pairs_shortest_paths` |
| `start` | index of the base camp, default 0 |
| `return_to_base` | whether the crew returns to `start` at wrap, default False |
| `region_cap` | maximum locations per region, default 10; no effect when the whole production already fits the exact solver |

`greedy_nearest_neighbor(dist_matrix, start, scene_indices=None)` additionally
takes an optional subset of nodes to schedule, and returns a `GreedyResult`:
`order`, `total_cost` and `cost_breakdown` (the cost of each step).

**Outputs.** The exact and partitioned solvers return a `ScheduleResult`:

| Field | Meaning |
|---|---|
| `order` | the shooting order as location indices; the base appears again at the end if the schedule is closed |
| `total_cost` | the sum of the legs |
| `leg_costs` | the cost of each individual move |
| `regions` | the regions used (partitioned solver only) |
| `states_settled` | (subset, last-location) states relaxed by the subset DP |
| `n_locations`, `closed` | the size of the instance, and whether it returns to base |

`solve()` wraps that in a `SolveReport`, which records how the answer was
reached: `mode`, the number of `regions`, the `guarantee` that holds for it,
and `elapsed_ms`.

### End to end

`python -m stripboard <production>` takes a production key — `la_la_land`,
`forrest_gump` or `tenet` — builds the road network, runs both layers, and
prints the schedule. For `la_la_land` it reports

```
[2] Scheduling layer
    regions   : 9
    guarantee : optimal within and between regions; the partition is
                the only approximation
    solved in : 6.6 ms
```

followed by the 26-row shooting order and the cost of every move.

### Testing

The test suite (`pytest`, under `tests/`) checks each layer against an
independent reference rather than against its own earlier output.

**Geographic layer** (`tests/test_geo.py`):
- All-pairs Dijkstra matches an independent Floyd–Warshall on toy, benchmark
  and sparse graphs up to n = 40
- Path reconstruction re-walks `prev[]` and re-costs the path against the
  adjacency matrix; unreachable targets return an empty path
- On connected graphs of n = 10 to 100, every node is settled
  (`visited_count = n`)
- The weight formula, the elevation cap, matrix symmetry, and rejection of
  negative weights
- Every seeded generator is reproducible (same seed → identical matrix) and
  connected; every production's road network is connected

**Greedy** (`tests/test_scheduling.py`): never revisits a node, exhausts every
scene, and its step costs sum to its total.

**Exact solver** (`tests/test_subset_dp.py`). Cross-checked against exhaustive
enumeration of every permutation, which is obviously correct and obviously too
slow to use. Random symmetric cost matrices, every size from n = 2 to 8, forty
matrices per size, open and returning-to-base:

> **560 cases, 0 disagreements.**

Reconstructed orders are re-walked and re-costed against the matrix, so a
correct total paired with a wrong path cannot pass.

**Partitioned path, one region** (`tests/test_clustered_dp.py`). The same
comparison, but forcing the region machinery to run rather than
short-circuiting to the exact solver (`always_partition=True`), sizes n = 2 to
9, twenty-five matrices per size, additionally requiring the reconstructed
order to equal the subset DP's:

> **400 cases, 0 disagreements.**

This checks the claim in Section 3.1: with one region the partitioned method
reproduces the exact optimum, so plain subset DP really is its degenerate case.

**Partitioned path, several regions.** A single region never reaches the
entry/exit table, the region-level transitions or the chain reconstruction, the
parts specific to this method. Those are checked against an oracle that
enumerates every region order together with every route within every region,
sizes n = 4 to 8 with caps of 2 and 3, producing between 2 and 7 regions:

> **200 cases, 0 disagreements.**

**Validity on real data.** Every schedule produced for the three productions is
checked to be a permutation of all locations whose leg costs sum to the
reported total.

---

## 6. Results

All timings are wall-clock and machine-dependent; costs and counts are
deterministic. `python -m experiments.reproduce` regenerates every table below
from Section 6.2 onward, and `python -m experiments.benchmark` the runtime
table in 6.1.

### 6.1 The geographic layer on the 8-scene benchmark

**All-pairs cost matrix.** Compare with the raw adjacency matrix in Section
4.1: every `0.00` off the diagonal has become a routed cost.

```
     Node   0      1      2      3      4      5      6      7
        0   0.00  14.55  52.89  28.61  12.68  42.51  22.01  24.27
        1  14.55   0.00  38.34  14.06  27.23  27.96   7.46   9.72
        2  52.89  38.34   0.00  35.17  48.01  10.38  30.88  36.54
        3  28.61  14.06  35.17   0.00  41.29  24.79  21.52  23.73
        4  12.68  27.23  48.01  41.29   0.00  37.63  34.69  36.95
        5  42.51  27.96  10.38  24.79  37.63   0.00  20.50  37.68
        6  22.01   7.46  30.88  21.52  34.69  20.50   0.00  17.18
        7  24.27   9.72  36.54  23.73  36.95  37.68  17.18   0.00
```

**Single-source statistics** (source = 0):

| Metric | Value |
|---|---|
| Nodes visited | 8 |
| Heap pushes | 12 |
| Wall-clock time | 0.014 ms |
| Path 0 → 7 | [0, 1, 7], cost = 24.27 |

**Greedy schedule:**

```
  Step 0: downtown_plaza    [START — basecamp]
  Step 1: desert_dunes      +12.68  (nearest from downtown)
  Step 2: forest_trail      +27.23
  Step 3: river_crossing    +7.46   (nearest forest neighbor)
  Step 4: cliff_edge        +17.18
  Step 5: coastal_cove      +23.73
  Step 6: ancient_ruins     +24.79
  Step 7: mountain_peak     +10.38
  ─────────────────────────────────
  TOTAL COST: 123.45
```

**Runtime** on seeded sparse graphs (edge probability 0.05):

| Algorithm | n=10 | n=50 | n=100 | n=500 |
|---|---|---|---|---|
| Dijkstra (all-pairs) | 0.09 ms | 4.25 ms | 29.35 ms | 3,690.40 ms |
| Greedy NN | 0.02 ms | 0.06 ms | 0.21 ms | 4.77 ms |

Dijkstra all-pairs grows roughly quadratically-to-cubically with n (it reruns
single-source Dijkstra from every node), while greedy stays fast since it only
does a single O(k²) pass once the cost matrix already exists.

| Figure | File |
|---|---|
| Runtime curve (log-log) | `figures/runtime_curves.png` |
| Adjacency heatmaps | `figures/film_benchmark_{6,8,12}_heatmap.png` |

### 6.2 What the exact solver reaches

| Locations | Time to prove optimal | DP states settled |
|---|---|---|
| 15 | 0.11 s | 114,689 |
| 16 | 0.26 s | 245,761 |
| 17 | 0.59 s | 524,289 |
| 18 | 1.32 s | 1,114,113 |
| 19 | 2.93 s | 2,359,297 |
| **20** | **6.58 s** | **4,980,737** |
| 21 | 14.08 s | 10,485,761 |
| 22 | 32.40 s | 22,020,097 |

States settled double with each added location, and the measured runtime
follows: the empirical curve is a straight line on a log scale, which is what
`2ⁿ` looks like (`figures/exact_scaling.png`). Memory is what runs out: peak
resident memory is 200 MB at n = 20, 398 MB at n = 21 and 812 MB at n = 22,
each measured in its own process (measuring several sizes in one process
reports the running peak, not the peak of each). Those figures track the
allocation the DP asks for, 9·n·2ⁿ bytes. The limit is therefore set at 20.

### 6.3 How far the greedy draft lands from the optimum

Running the synthetic benchmarks through the exact solver answers what neither
layer can answer alone:

| Benchmark | Greedy draft | Proven optimum | Gap |
|---|---|---|---|
| 6 scenes | 332.81 | 304.88 | +9.16% |
| 8 scenes | 123.45 | 123.45 | **+0.00%** |
| 12 scenes | 179.47 | 140.07 | +28.13% |

The draft happens to be optimal on the 8-scene benchmark and 28% off on the
12-scene one. Nearest-neighbour gives no guarantee either way, and only an
exact solver distinguishes the two cases.

### 6.4 Scheduling the three productions

| Production | Greedy draft | This method | Saving | Time | Regions |
|---|---|---|---|---|---|
| La La Land | 235.86 | **210.68** | **10.68%** | 6.1 ms | 9 |
| Forrest Gump | 12,927.26 | **12,124.09** | **6.21%** | 15.9 ms | 10 |
| Tenet | 36,203.99 | **34,283.38** | **5.30%** | 38.7 ms | 11 |

*La La Land* and *Tenet* move between places the fewest times possible, 11 of 11
and 10 of 10. *Forrest Gump* takes 16 against a floor of 15, and the extra one is
a labelling artefact rather than a detour: Los Angeles, Santa Monica and Monterey
Park are three labels for one metropolitan block, and the schedule interleaves
them as a crew would.

On *La La Land* the greedy draft leaves West Hills, the furthest north-west
location, for last, and pays **75.5** to cross the county back to it from Long
Beach. That single leg is more than twice any other in its
schedule. This method takes West Hills second as a place (sixth location
overall) straight out of the Burbank base, for **33.7**. It then works east
across Hollywood, Griffith Park and Pasadena before turning south to Downtown,
Hermosa Beach and Long Beach, and never returns to a place it has left. Its
most expensive leg is that same 33.7 (`figures/routes_la_la_land.png`).

### 6.5 What the partition costs

On the largest subset of each production the solver can
still prove outright (20 locations), the same method is run twice, once with
the partition forced and once without, and both are measured against that proven
optimum:

| Production | Proven optimum | Time to prove | Partitioned | Gap | Greedy draft | Gap |
|---|---|---|---|---|---|---|
| La La Land | 117.16 | 6.39 s | 117.16 | **+0.00%** | 121.84 | +3.99% |
| Forrest Gump | 6,708.20 | 6.40 s | 6,708.20 | **+0.00%** | 6,715.96 | +0.12% |
| Tenet | 13,600.76 | 6.41 s | 13,600.76 | **+0.00%** | 13,602.93 | +0.02% |

**Partitioning found the proven optimum on all three, in milliseconds against
six and a half seconds** (`figures/optimality_gap.png`). Scope matters: these
are the largest subsets that can still be proved, and the productions at full
size are past that point.

### 6.6 What the partition discovers

The partition reads only the cost matrix, with no place names or coordinates,
and still reproduces the real geography at every scale:

| Production | Regions found | Matching a single real place |
|---|---|---|
| La La Land | 9 | 7 |
| Forrest Gump | 10 | 7 |
| Tenet | 11 | **10** |

On *La La Land* the method
merges Hollywood, West Hollywood and Midtown into one block, three adjacent
districts, and merges Pasadena with South Pasadena, which share a border. On
*Forrest Gump* it merges Walterboro, Yemassee, McPhersonville and Varnville,
four rural South Carolina towns all within 39 km of one another, and merges Los
Angeles with Santa Monica and Monterey Park, which is what a production would
call one block regardless of the municipal lines between them.
**Most of the time the partition disagrees with the labels because the labels
are finer than the geography warrants.** Not always: its third *Forrest Gump*
merge joins Cut Bank to Glacier National Park, 82–89 km apart. Those are the
only two Montana entries on a map that otherwise runs from Maine to California,
so nothing nearer exists to group them with. That is the limit of reading
regions off distances alone.

### 6.7 Why the pipeline needs both layers

| Production | Pairs | No direct road | Share | Detour is cheaper |
|---|---|---|---|---|
| La La Land | 325 | 275 | **84.6%** | 2 |
| Forrest Gump | 435 | 364 | **83.7%** | 16 |
| Tenet | 561 | 458 | **81.6%** | 16 |

Most of the scheduling layer's input exists only because shortest paths were
computed for it. Getting from Chippewa Square in Savannah to Marks Hall on the
USC campus, for instance, is a seven-hop route through McPhersonville,
Varnville, Grandfather Mountain, Monument Valley, Flagstaff and Monterey Park.

### 6.8 What the terrain weighting contributes

Each production was scheduled twice, once on the geographic layer's terrain and
elevation weights and once on raw kilometres, and both orders priced with the
real weights:

| Production | Cost of ignoring terrain | Non-urban locations | Distance spread |
|---|---|---|---|
| La La Land | +0.00% | 23% | 628× |
| Forrest Gump | +0.00% | 50% | 15,458× |
| Tenet | **+0.54%** | 41% | 136,459× |

The terrain multiplier tops out at 2×, while distances on these maps span five
orders of magnitude, so **terrain can only decide between locations that are
already close.** On the two US productions the schedule costs the same with and
without it: the orders differ, but not in a way that changes the total. Only
*Tenet*, with four locations on Amalfi cliffs and two in desert, differs at all,
by half a percent.

### 6.9 Choosing the region size cap

Larger regions are solved more exactly, but *finish this region before moving
on* is a constraint, so enlarging regions tightens it. The two effects pull
against each other:

| Cap | La La Land | Forrest Gump | Tenet |
|---|---|---|---|
| ≤ 2 | *past the region limit* | *past the region limit* | *past the region limit* |
| 3–4 | 210.68 | *past the region limit* | *past the region limit* |
| 5 | 210.68 | 12,131.85 | *past the region limit* |
| 6 | 210.68 | 12,131.85 | 34,284.59 |
| 7–9 | 210.68 | 12,124.09 | 34,284.59 |
| **10** | **210.68** | **12,124.09** | **34,283.38** |
| 11 | 210.68 | 12,124.09 | 34,283.38 |
| 12 | 217.73 | 12,124.09 | 34,283.38 |

No single cap is best everywhere. Below 6, *Tenet* needs more regions than the
region-level DP allows and cannot be partitioned at all; *Forrest Gump* needs
at least 5 and *La La Land* at least 3. At 12, *La La Land* costs 3.3% more,
because a cap that large stops splitting a block the schedule would rather
interleave. Caps 7 through 11 form a flat band in which all three productions
sit at their own best, or within 0.01% of it (`figures/region_cap.png`). **10
is inside that band and is the smallest cap at which the largest production
reaches its best cost**, so it is the default.

### 6.10 The upper limit

`region_cap × MAX_REGIONS` puts a hard ceiling of **130 locations** on the
partition, but the count alone does not decide it — the geometry does. Thirteen
tight clusters of ten succeed all the way to 130 and fail at 131; eight
clusters succeed to 80; uniformly scattered points, with no cluster structure
for the tree to find, first fail around n = 40 and essentially always fail by
80 — the exact onset moves with the seed. When no partition can be formed the
code falls back to the greedy draft, and it does so because the partition
actually failed rather than because a location count was exceeded — an earlier
version predicted from the count and crashed in the 126–130 range. The largest
production here is 34 locations, below every one of these thresholds.

---

## 7. Conclusion

### Answer to the research question

The shooting order that minimises transition cost can be found in two steps,
and each step needs its own layer. Dijkstra turns a terrain-weighted road
network into the true cost of every move — and on real productions 81–85% of
those moves have no direct road, so this step is not optional. The scheduling
layer then orders the locations.

For a production of up to 20 locations, the optimal order can be computed and
**proved** optimal in about six seconds — verified against exhaustive search on
560 cases with no disagreement. Beyond that size the problem is made tractable
by one assumption, block shooting, which is what productions do anyway; every
step after that assumption remains exact, up to a ceiling of 130 locations set
by the region cap and region count.

A greedy nearest-neighbour draft is fast but unreliable: on the synthetic
benchmarks it landed anywhere from 0% to 28% above the proven optimum. On the
three real productions the partitioned method's schedules cost **5.3% to 10.7%
less** than that draft, and on the largest subsets where a true optimum could
still be proved, partitioning **found that optimum on all three** — in
milliseconds against six and a half seconds.

### Weaknesses and limitations

1. **The partition is unverified above 20 locations.** Its cost can only be
   measured where an exact optimum is still computable. On larger productions —
   which is every production studied here at full size — the reported saving is
   against the greedy draft, not against a proven optimum, because no proven
   optimum exists at that size.

2. **The region count comes from a heuristic rule.** Cutting at the widest
   ratio gap in the MST's sorted edge weights works well on maps with real
   clusters, but it is a rule read off the data, not a method with a
   guarantee. A production whose locations are spread evenly rather than
   clustered would have no such gap to find, and the partition would have no
   structure to exploit.

3. **Distances are straight lines, not roads.** Synthetic benchmarks use
   formula-generated weights, and the real productions use great-circle
   distance between real coordinates. Road closures, one-way streets, ferry and
   flight schedules and seasonal conditions would need a real map API.

4. **Location data is reconstructed, not official.** The lists come from
   enthusiast location guides rather than production paperwork, and the real
   shooting orders are unpublished, so these schedules cannot be compared with
   what the crews actually did.

5. **Only transition cost is modelled.** Crew and cast availability, permit
   windows, day-for-night requirements and story continuity all constrain a real
   schedule and none of them appear here. Section 1 shows that per-location
   shooting cost is genuinely irrelevant to the ordering; these other
   constraints are not, and their absence is a modelling choice, not a proof.

6. **The graph is undirected.** Real transit is often asymmetric — uphill
   against downhill, one-way roads, flight schedules — which the cost matrix
   cannot express as it stands.

7. **All-pairs memory.** The n × n cost matrix is ~781 MB at n = 10,000,
   impractical for very large location sets.

8. **Single crew unit.** Multi-unit productions (simultaneous second units)
   require a parallel scheduling formulation.

### Future work

- **Precedence constraints from the script.** Some scenes must be shot before
  others: a building is destroyed, an actor's appearance changes, snow melts.
  These are exactly the constraints a map cannot know and a script can. They fit
  the subset DP directly — an ordering is legal only if a subset is closed under
  its predecessors — and, counter-intuitively, would make the DP *faster* by
  making many subsets illegal.

- **A guaranteed baseline.** CLRS §35.2 gives a 2-approximation for metric TSP
  built from a minimum spanning tree, and the MST is already computed here. The
  cost matrix satisfies the triangle inequality by construction, since it holds
  shortest paths, so the guarantee applies. That would add the one thing the
  scheduling layer currently lacks at large sizes: a provable bound rather than
  a measured gap.

- **Verifying the partition at scale** by using that bound, or by branch and
  bound, to establish an optimum for productions of 25–40 locations where the
  subset DP cannot reach.

- **Cheaper improvements on the draft.** 2-opt local search after greedy, or
  simulated annealing, as middle ground between the draft and the exact solver.

- **Real road data.** Replace great-circle distance with an OpenStreetMap road
  network and elevation API, and use A\* with a geographic heuristic for
  single-source pathfinding on graphs too large for all-pairs Dijkstra.

- **Multi-unit crews.** Partition scenes across units and minimise makespan.
