"""Time-dependent traffic and traffic-aware routing.

Each edge's travel time at hour ``h`` is

    t_e(h) = freeflow_e * global_scale * congestion[zone(e), h]

``congestion`` is a (Z, 24) table. By default it is a hand-built profile with
AM / PM rush-hour peaks (stronger near transit hubs). When a Google Maps
calibration file exists (see :mod:`taxi_rl.gmaps`), the table is replaced by
ratios measured from Google's traffic-aware ETAs.

:class:`Router` computes all-pairs shortest paths *under the hour's traffic*
with SciPy's C Dijkstra and caches one matrix per hour, so the route between
two points really changes between 3 AM and 6 PM.
"""

from __future__ import annotations

import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .graph import DATA_DIR, RoadGraph

CALIBRATION_PATH = DATA_DIR / "traffic_calibration.json"


def kmeans(points: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Tiny deterministic k-means (labels, centers) — avoids a sklearn dependency."""
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=k, replace=False)].copy()
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(iters):
        d = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        new = np.array(
            [points[labels == c].mean(0) if np.any(labels == c) else centers[c] for c in range(k)]
        )
        if np.allclose(new, centers):
            break
        centers = new
    # Stable ordering: sort zones south->north, then west->east
    order = np.lexsort((centers[:, 0], np.round(centers[:, 1], 1)))
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    return remap[labels], centers[order]


@dataclass
class Zones:
    labels: np.ndarray  # (N,) zone of each node
    centers_xy: np.ndarray  # (Z, 2) km
    center_nodes: np.ndarray  # (Z,) node closest to each zone center

    @property
    def n(self) -> int:
        return int(self.centers_xy.shape[0])

    @classmethod
    def build(cls, graph: RoadGraph, n_zones: int = 8, seed: int = 0) -> Zones:
        xy = graph.xy_km()
        labels, centers = kmeans(xy, n_zones, seed=seed)
        center_nodes = np.array(
            [int(np.argmin(((xy - c) ** 2).sum(1))) for c in centers], dtype=np.int64
        )
        return cls(labels=labels, centers_xy=centers, center_nodes=center_nodes)


def default_congestion_profile(n_zones: int, hub_weight: np.ndarray | None = None) -> np.ndarray:
    """(Z, 24) multiplicative congestion: ~1.0 at night, ~1.9 at PM peak."""
    h = np.arange(24, dtype=np.float64)
    base = (
        1.0
        + 0.55 * np.exp(-((h - 8.5) ** 2) / 2.5)  # AM peak
        + 0.25 * np.exp(-((h - 13.0) ** 2) / 6.0)  # midday
        + 0.70 * np.exp(-((h - 17.5) ** 2) / 3.0)  # PM peak
        + 0.10 * np.exp(-((h - 22.0) ** 2) / 2.0)  # theatre let-out
    )
    if hub_weight is None:
        hub_weight = np.ones(n_zones)
    hub_weight = np.asarray(hub_weight, dtype=np.float64)
    return 1.0 + (base[None, :] - 1.0) * hub_weight[:, None]


@dataclass
class TrafficModel:
    congestion: np.ndarray  # (Z, 24)
    global_scale: float = 1.0
    trip_noise_sigma: float = 0.08  # lognormal noise applied per realised trip
    source: str = "default-profile"
    meta: dict = field(default_factory=dict)

    def multiplier(self, zone: np.ndarray | int, hour: float) -> np.ndarray:
        """Linearly interpolated multiplier for fractional hours."""
        h0 = int(np.floor(hour)) % 24
        h1 = (h0 + 1) % 24
        w = hour - np.floor(hour)
        c = self.congestion
        return self.global_scale * ((1 - w) * c[zone, h0] + w * c[zone, h1])

    @classmethod
    def default(cls, zones: Zones, graph: RoadGraph) -> TrafficModel:
        # Zones closer to the centroid of the network (Midtown core) get heavier traffic.
        xy = graph.xy_km()
        mid = xy.mean(0)
        dist = np.linalg.norm(zones.centers_xy - mid, axis=1)
        hub = 1.25 - 0.5 * (dist / (dist.max() + 1e-9))
        return cls(congestion=default_congestion_profile(zones.n, hub))

    @classmethod
    def load(cls, path: Path | str = CALIBRATION_PATH) -> TrafficModel:
        d = json.loads(Path(path).read_text())
        return cls(
            congestion=np.asarray(d["congestion"], dtype=np.float64),
            global_scale=float(d.get("global_scale", 1.0)),
            trip_noise_sigma=float(d.get("trip_noise_sigma", 0.08)),
            source=d.get("source", str(path)),
            meta=d.get("meta", {}),
        )

    def save(self, path: Path | str = CALIBRATION_PATH) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "congestion": np.round(self.congestion, 4).tolist(),
                    "global_scale": self.global_scale,
                    "trip_noise_sigma": self.trip_noise_sigma,
                    "source": self.source,
                    "meta": self.meta,
                },
                indent=2,
            )
        )

    @classmethod
    def auto(cls, zones: Zones, graph: RoadGraph, use_calibration: bool = True) -> TrafficModel:
        if use_calibration and CALIBRATION_PATH.exists():
            tm = cls.load(CALIBRATION_PATH)
            if tm.congestion.shape[0] == zones.n:
                return tm
        return cls.default(zones, graph)


class Router:
    """Traffic-aware all-pairs shortest paths, cached per hour."""

    def __init__(self, graph: RoadGraph, zones: Zones, traffic: TrafficModel, cache_hours: int = 6):
        self.g = graph
        self.zones = zones
        self.traffic = traffic
        # An edge belongs to the zone of its tail node.
        self.edge_zone = zones.labels[graph.edge_u]
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_hours = cache_hours
        self._edge_index = {
            (int(u), int(v)): i for i, (u, v) in enumerate(zip(graph.edge_u, graph.edge_v))
        }

    def edge_times(self, hour: float) -> np.ndarray:
        return self.g.edge_freeflow_s * self.traffic.multiplier(self.edge_zone, hour)

    def _tables(self, hour: float) -> tuple[np.ndarray, np.ndarray]:
        key = int(np.floor(hour)) % 24
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        w = self.edge_times(key + 0.5)
        n = self.g.n_nodes
        mat = csr_matrix((w, (self.g.edge_u, self.g.edge_v)), shape=(n, n))
        dist, pred = dijkstra(mat, directed=True, return_predecessors=True)
        tables = (dist.astype(np.float32), pred.astype(np.int32))
        self._cache[key] = tables
        if len(self._cache) > self._cache_hours:
            self._cache.popitem(last=False)
        return tables

    def travel_time(self, u: int, v: int, hour: float) -> float:
        """Expected travel time in seconds (no noise)."""
        return float(self._tables(hour)[0][u, v])

    def times_from(self, u: int, hour: float) -> np.ndarray:
        return self._tables(hour)[0][u]

    def times_to(self, v: int, hour: float) -> np.ndarray:
        return self._tables(hour)[0][:, v]

    def path(self, u: int, v: int, hour: float) -> list[int]:
        pred = self._tables(hour)[1]
        out = [v]
        while out[-1] != u:
            p = int(pred[u, out[-1]])
            if p < 0:
                raise ValueError(f"No path {u}->{v}")
            out.append(p)
        return out[::-1]

    def path_length_m(self, path: list[int]) -> float:
        return float(
            sum(self.g.edge_length_m[self._edge_index[(a, b)]] for a, b in itertools.pairwise(path))
        )

    def path_edges(self, path: list[int]) -> list[int]:
        return [self._edge_index[(a, b)] for a, b in itertools.pairwise(path)]
