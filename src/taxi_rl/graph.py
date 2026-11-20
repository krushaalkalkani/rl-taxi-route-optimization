"""Road-network loading.

The simulator runs on a real street graph pulled from OpenStreetMap (via OSMnx)
and cached as GraphML in ``data/``. For tests / CI (no network) a synthetic
Manhattan-style grid with one-way avenues is available.

Everything downstream works on a compact integer-indexed representation
(:class:`RoadGraph`) so shortest paths can be computed with SciPy's C Dijkstra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_GRAPHML = DATA_DIR / "midtown.graphml"

# (west, south, east, north) — Midtown Manhattan, ~3.4 km x 3.3 km
MIDTOWN_BBOX = (-74.005, 40.740, -73.965, 40.770)


@dataclass
class RoadGraph:
    """Integer-indexed directed road graph."""

    lat: np.ndarray  # (N,)
    lon: np.ndarray  # (N,)
    edge_u: np.ndarray  # (E,)
    edge_v: np.ndarray  # (E,)
    edge_length_m: np.ndarray  # (E,)
    edge_freeflow_s: np.ndarray  # (E,) free-flow traversal time
    edge_geometry: list[list[tuple[float, float]]]  # per-edge [(lat, lon), ...]
    name: str = "graph"

    @property
    def n_nodes(self) -> int:
        return int(self.lat.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_u.shape[0])

    def xy_km(self) -> np.ndarray:
        """Local equirectangular projection in km, origin at the SW corner."""
        lat0 = float(self.lat.min())
        lon0 = float(self.lon.min())
        kx = 111.32 * math.cos(math.radians(float(self.lat.mean())))
        return np.stack([(self.lon - lon0) * kx, (self.lat - lat0) * 110.57], axis=1)

    def nearest_node(self, lat: float, lon: float) -> int:
        kx = math.cos(math.radians(lat))
        d2 = ((self.lat - lat) * 110.57) ** 2 + ((self.lon - lon) * 111.32 * kx) ** 2
        return int(np.argmin(d2))


def _largest_scc(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    nodes = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def from_networkx(G: nx.MultiDiGraph, name: str = "graph") -> RoadGraph:
    """Convert an OSMnx-style MultiDiGraph into a :class:`RoadGraph`.

    Parallel edges are collapsed to the fastest one; only the largest strongly
    connected component is kept so every (origin, destination) pair is routable.
    """
    G = _largest_scc(G)
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    lat = np.array([G.nodes[n]["y"] for n in nodes], dtype=np.float64)
    lon = np.array([G.nodes[n]["x"] for n in nodes], dtype=np.float64)

    best: dict[tuple[int, int], tuple[float, float, list[tuple[float, float]]]] = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        length = float(data.get("length", 0.0)) or 1.0
        tt = data.get("travel_time")
        if tt is None:
            speed = float(data.get("speed_kph", 25.0))
            tt = length / (speed / 3.6)
        tt = float(tt)
        geom = data.get("geometry")
        if geom is not None:
            pts = [(float(y), float(x)) for x, y in geom.coords]
        else:
            pts = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
        key = (idx[u], idx[v])
        if key not in best or tt < best[key][1]:
            best[key] = (length, tt, pts)

    keys = sorted(best)
    return RoadGraph(
        lat=lat,
        lon=lon,
        edge_u=np.array([k[0] for k in keys], dtype=np.int32),
        edge_v=np.array([k[1] for k in keys], dtype=np.int32),
        edge_length_m=np.array([best[k][0] for k in keys], dtype=np.float64),
        edge_freeflow_s=np.array([best[k][1] for k in keys], dtype=np.float64),
        edge_geometry=[best[k][2] for k in keys],
        name=name,
    )


def download_osm(bbox=MIDTOWN_BBOX, path: Path = DEFAULT_GRAPHML) -> Path:
    """Download a drivable street network from OpenStreetMap and cache it."""
    import osmnx as ox

    mirrors = [
        "https://overpass-api.de/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
        "https://overpass.kumi.systems/api",
    ]
    last: Exception | None = None
    for url in mirrors:
        ox.settings.overpass_url = url
        try:
            G = ox.graph_from_bbox(bbox, network_type="drive")
            break
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - network dependent
            last = exc
    else:  # pragma: no cover
        raise RuntimeError(f"All Overpass mirrors failed: {last}")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    return path


def load_osm(path: Path | str = DEFAULT_GRAPHML) -> RoadGraph:
    import osmnx as ox

    path = Path(path)
    if not path.exists():
        download_osm(path=path)
    G = ox.load_graphml(path)
    return from_networkx(G, name=path.stem)


def synthetic_grid(
    n_streets: int = 14,
    n_avenues: int = 6,
    block_m: tuple[float, float] = (80.0, 260.0),
    origin: tuple[float, float] = (40.748, -73.992),
) -> RoadGraph:
    """Manhattan-like grid: two-way cross streets alternate one-way, avenues
    alternate north/south-bound. Used for tests and offline demos."""
    G = nx.MultiDiGraph()
    lat0, lon0 = origin
    kx = 111_320 * math.cos(math.radians(lat0))
    for i in range(n_streets):
        for j in range(n_avenues):
            y = lat0 + i * block_m[0] / 110_570
            x = lon0 + j * block_m[1] / kx
            G.add_node((i, j), y=y, x=x)

    def add(a, b, length, speed_kph):
        G.add_edge(a, b, length=length, travel_time=length / (speed_kph / 3.6))

    for i in range(n_streets):
        for j in range(n_avenues):
            if i + 1 < n_streets:  # avenue segment (north-south)
                if j % 2 == 0:
                    add((i, j), (i + 1, j), block_m[0], 35)
                else:
                    add((i + 1, j), (i, j), block_m[0], 35)
            if j + 1 < n_avenues:  # cross street (east-west)
                if i % 4 == 0:  # every 4th is a two-way major street
                    add((i, j), (i, j + 1), block_m[1], 30)
                    add((i, j + 1), (i, j), block_m[1], 30)
                elif i % 2 == 0:
                    add((i, j), (i, j + 1), block_m[1], 25)
                else:
                    add((i, j + 1), (i, j), block_m[1], 25)
    return from_networkx(G, name="synthetic_grid")


def load_graph(source: str = "osm") -> RoadGraph:
    """``"osm"`` (cached Midtown Manhattan), ``"grid"``, or a path to GraphML."""
    if source == "grid":
        return synthetic_grid()
    if source == "osm":
        return load_osm(DEFAULT_GRAPHML)
    return load_osm(source)
