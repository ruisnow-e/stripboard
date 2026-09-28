"""
graph.py — Weighted graph schema and spatial modeling for film location scheduling.

Node schema: location name, terrain type, elevation, basecamp flag.
Edge schema: distance, elevation change, terrain difficulty → combined weight.
Adjacency matrix interface consumed by the shortest-path step.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class TerrainType(Enum):
    """Terrain categories with associated travel difficulty multipliers."""
    URBAN = "urban"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    COASTAL = "coastal"
    DESERT = "desert"


# Terrain multipliers used in weight formula:
#   w = distance * (1 + 0.3 * elevation_factor) * terrain_multiplier
TERRAIN_MULTIPLIER = {
    TerrainType.URBAN:    1.0,
    TerrainType.FOREST:   1.4,
    TerrainType.MOUNTAIN: 2.0,
    TerrainType.COASTAL:  1.2,
    TerrainType.DESERT:   1.6,
}


@dataclass
class Node:
    """
    Represents a filming location (graph node).

    Attributes:
        id          : Unique integer index (0-based, matches matrix row/col).
        name        : Human-readable location name (e.g. 'mountain_peak').
        terrain_type: One of the TerrainType enum values.
        elevation_m : Elevation above sea level in metres.
        is_basecamp : True if this node is the production basecamp / start node.
    """
    id: int
    name: str
    terrain_type: TerrainType
    elevation_m: float
    is_basecamp: bool = False

    def terrain_multiplier(self) -> float:
        """Return the travel-cost multiplier for this node's terrain type."""
        return TERRAIN_MULTIPLIER[self.terrain_type]

    def __repr__(self) -> str:
        flag = " [basecamp]" if self.is_basecamp else ""
        return (f"Node({self.id}, '{self.name}', {self.terrain_type.value}, "
                f"{self.elevation_m}m{flag})")


@dataclass
class Edge:
    """
    Represents a directed edge between two filming locations.

    Attributes:
        src              : Source node index.
        dst              : Destination node index.
        distance         : Raw geographic distance (arbitrary units, e.g. km).
        elevation_change : Absolute elevation difference between endpoints (m).
        terrain_difficulty: Composite terrain multiplier (average of endpoints).
        weight           : Combined edge weight used by algorithms.
    """
    src: int
    dst: int
    distance: float
    elevation_change: float
    terrain_difficulty: float
    weight: float

    @staticmethod
    def compute_weight(distance: float, elevation_change: float,
                       terrain_difficulty: float) -> float:
        """
        Compute edge weight using the project formula:
            w = distance * (1 + 0.3 * elevation_factor) * terrain_multiplier

        The elevation_factor is elevation_change normalised to a [0, 1] range by
        assuming a reference maximum of 3000 m; the terrain_difficulty is already
        the combined multiplier passed from the caller.
        """
        elevation_factor = min(elevation_change / 3000.0, 1.0)
        return distance * (1.0 + 0.3 * elevation_factor) * terrain_difficulty

    @classmethod
    def from_nodes(cls, src_node: Node, dst_node: Node,
                   distance: float) -> "Edge":
        """
        Construct an Edge between two Node objects.

        The terrain_difficulty is the average of the two endpoints' multipliers,
        and the elevation_change is the absolute altitude difference.
        """
        elev_change = abs(src_node.elevation_m - dst_node.elevation_m)
        terrain_diff = (src_node.terrain_multiplier() +
                        dst_node.terrain_multiplier()) / 2.0
        w = Edge.compute_weight(distance, elev_change, terrain_diff)
        return cls(
            src=src_node.id,
            dst=dst_node.id,
            distance=distance,
            elevation_change=elev_change,
            terrain_difficulty=terrain_diff,
            weight=w,
        )


class SpatialGraph:
    """
    Undirected weighted graph of filming locations stored as an adjacency matrix.

    The adjacency matrix uses the convention:
        matrix[i][j] == 0  →  no edge between i and j
        matrix[i][j] >  0  →  edge weight

    Dijkstra reads this format; the schedulers read its all-pairs output.
    """

    def __init__(self) -> None:
        """Initialise an empty graph."""
        self.nodes: List[Node] = []
        self._matrix: List[List[float]] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def load_from_matrix(cls, matrix: List[List[float]],
                         nodes: List[Node]) -> "SpatialGraph":
        """
        Build a SpatialGraph from an existing adjacency matrix and node list.

        Args:
            matrix: 2-D list of floats (n x n).  0 = no edge.
            nodes : List of Node objects; len(nodes) must equal len(matrix).

        Returns:
            A fully initialised SpatialGraph.

        Raises:
            ValueError: If dimensions are inconsistent.
        """
        n = len(matrix)
        if len(nodes) != n:
            raise ValueError(
                f"Matrix size {n} does not match nodes count {len(nodes)}.")
        for i, row in enumerate(matrix):
            if len(row) != n:
                raise ValueError(
                    f"Matrix row {i} has length {len(row)}, expected {n}.")

        g = cls()
        g.nodes = list(nodes)
        # Deep copy so external mutations don't affect the graph
        g._matrix = [[float(matrix[i][j]) for j in range(n)]
                     for i in range(n)]
        return g

    def add_node(self, node: Node) -> None:
        """
        Append a new node, expanding the adjacency matrix with zero-filled rows/cols.

        The node's id is updated to match its index in self.nodes.
        """
        idx = len(self.nodes)
        node.id = idx
        self.nodes.append(node)
        # Expand existing rows
        for row in self._matrix:
            row.append(0.0)
        # Add new row
        self._matrix.append([0.0] * (idx + 1))

    def set_edge(self, i: int, j: int, w: float) -> None:
        """
        Set (or overwrite) the undirected edge weight between nodes i and j.

        Args:
            i: Source node index.
            j: Destination node index.
            w: Edge weight (must be > 0; use remove_edge to delete an edge).
        """
        if w < 0:
            raise ValueError(f"Edge weight must be non-negative, got {w}.")
        self._matrix[i][j] = w
        self._matrix[j][i] = w

    def remove_edge(self, i: int, j: int) -> None:
        """Remove the edge between nodes i and j (sets weight to 0)."""
        self._matrix[i][j] = 0.0
        self._matrix[j][i] = 0.0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def weight(self, i: int, j: int) -> float:
        """
        Return the edge weight between nodes i and j.

        Returns 0.0 if the nodes are not connected (no edge).
        """
        return self._matrix[i][j]

    def neighbors(self, i: int) -> List[Tuple[int, float]]:
        """
        Return the list of (neighbour_index, weight) pairs for node i.

        Only connected neighbours (weight > 0) are included.
        """
        row = self._matrix[i]
        return [(j, row[j]) for j in range(len(row)) if row[j] > 0.0]

    def to_matrix(self) -> List[List[float]]:
        """
        Export the adjacency matrix as a 2-D list of floats.

        Returns a deep copy so the caller may modify it freely.
        """
        return [[self._matrix[i][j] for j in range(self.n)]
                for i in range(self.n)]

    def basecamp(self) -> Optional[Node]:
        """Return the first node marked as basecamp, or None."""
        for node in self.nodes:
            if node.is_basecamp:
                return node
        return None

    def is_connected(self) -> bool:
        """
        Return True if the graph is connected (every node is reachable from node 0).

        Uses iterative BFS.
        """
        if self.n == 0:
            return True
        visited = [False] * self.n
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop()
            for v, _ in self.neighbors(u):
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        return count == self.n

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SpatialGraph(n={self.n}, nodes={[nd.name for nd in self.nodes]})"

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the graph."""
        edge_count = sum(
            1 for i in range(self.n) for j in range(i + 1, self.n)
            if self._matrix[i][j] > 0
        )
        lines = [
            f"SpatialGraph: {self.n} nodes, {edge_count} edges",
            "Nodes:",
        ]
        for nd in self.nodes:
            lines.append(f"  {nd}")
        return "\n".join(lines)
