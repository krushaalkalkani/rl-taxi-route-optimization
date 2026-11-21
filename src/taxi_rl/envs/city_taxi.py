"""CityTaxi-v0 — a single taxi working a shift on a real road network.

This is a semi-MDP: each decision ("serve request k", "reposition to zone z",
"wait") takes a variable amount of simulated time, during which traffic and
demand keep evolving.

Action space  ``Discrete(n_slots + n_zones + 1)``
    0 .. n_slots-1                 serve the k-th closest open request
    n_slots .. n_slots+n_zones-1   drive empty to the centre of zone z
    last                           wait in place for 60 s

Invalid actions (empty slots, requests that cannot be reached before the rider
gives up, repositioning to the zone you're standing in) are exposed through
:meth:`action_masks` for ``sb3_contrib.MaskablePPO``.

Reward = (fare − driving cost) / 10, with a penalty when a rider cancels
because the taxi arrived too late. Airport riders take the taxi out of the
network for the outbound leg and the return (see :mod:`taxi_rl.demand`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..demand import AIRPORT, DemandModel, Request, fare_usd
from ..graph import RoadGraph, load_graph
from ..traffic import Router, TrafficModel, Zones

COST_PER_KM = 0.30  # fuel + wear, USD
CANCEL_PENALTY = 3.0  # USD-equivalent
WAIT_S = 60.0
REWARD_SCALE = 10.0


@dataclass
class World:
    graph: RoadGraph
    zones: Zones
    traffic: TrafficModel
    router: Router
    demand: DemandModel
    zone_intensity: np.ndarray  # (Z, 96) requests/min per 15-min bucket
    xy_norm: np.ndarray  # (N, 2) in [0, 1]
    exit_node: int  # where airport trips leave / re-enter the network

    def city_mult(self, hour: float) -> float:
        """Average congestion multiplier — drives the out-of-network airport legs."""
        return float(np.mean(self.traffic.multiplier(np.arange(self.zones.n), hour)))


@lru_cache(maxsize=4)
def build_world(graph_source: str = "osm", n_zones: int = 8, use_calibration: bool = True) -> World:
    graph = load_graph(graph_source)
    zones = Zones.build(graph, n_zones)
    traffic = TrafficModel.auto(zones, graph, use_calibration=use_calibration)
    router = Router(graph, zones, traffic, cache_hours=24)
    demand = DemandModel(graph)
    zi = np.zeros((zones.n, 96))
    for q in range(96):
        node_i = demand.node_pickup_intensity(q / 4.0)
        zi[:, q] = np.bincount(zones.labels, weights=node_i, minlength=zones.n)
    xy = graph.xy_km()
    xy_norm = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)
    exit_node = graph.nearest_node(*AIRPORT["exit"])
    return World(graph, zones, traffic, router, demand, zi, xy_norm, exit_node)


@dataclass
class Candidate:
    req: Request
    eta_s: float
    trip_s: float
    trip_m: float
    fare: float


@dataclass
class EpisodeStats:
    revenue: float = 0.0
    cost: float = 0.0
    served: int = 0
    cancelled: int = 0
    seen: int = 0
    km_total: float = 0.0
    km_empty: float = 0.0
    busy_s: float = 0.0
    log: list = field(default_factory=list)

    airport_trips: int = 0
    surge_trips: int = 0

    def summary(self, elapsed_s: float) -> dict:
        hours = elapsed_s / 3600.0
        return {
            "revenue": round(self.revenue, 2),
            "profit": round(self.revenue - self.cost, 2),
            "profit_per_hour": round((self.revenue - self.cost) / hours, 2),
            "served": self.served,
            "cancelled": self.cancelled,
            "requests_seen": self.seen,
            "km_total": round(self.km_total, 2),
            "km_empty": round(self.km_empty, 2),
            "empty_ratio": round(self.km_empty / max(self.km_total, 1e-9), 3),
            "utilization": round(self.busy_s / elapsed_s, 3),
            "airport_trips": self.airport_trips,
            "surge_trips": self.surge_trips,
            "hours": round(hours, 3),
        }


class CityTaxiEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}  # noqa: RUF012 (gymnasium convention)

    def __init__(
        self,
        graph_source: str = "osm",
        n_slots: int = 5,
        n_zones: int = 8,
        shift_hours: float = 4.0,
        start_hour: float | None = None,
        use_calibration: bool = True,
        record: bool = False,
        render_mode: str | None = None,
        opportunity_cost_per_hour: float = 0.0,
    ):
        super().__init__()
        # Average-reward shaping for the semi-MDP: every action is charged its duration times
        # a reference earning rate, so a trip only scores positive if it beats the going rate.
        # The shaped return is profit − λ·(shift time), which has the same optimum but far lower
        # variance. Evaluation always reports raw profit (``episode_stats``).
        self.opp_cost = opportunity_cost_per_hour
        self.w = build_world(graph_source, n_zones, use_calibration)
        self.n_slots = n_slots
        self.n_zones = self.w.zones.n
        self.shift_s = shift_hours * 3600.0
        self.fixed_start = start_hour
        self.record = record
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(n_slots + self.n_zones + 1)
        self.obs_dim = 7 + 8 * n_slots + 5 * self.n_zones
        self.observation_space = spaces.Box(-1.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)
        self._zi_max = float(self.w.zone_intensity.max())

    # ------------------------------------------------------------------ time
    def hour_of(self, t: float) -> float:
        return (self.start_hour + t / 3600.0) % 24.0

    def _zone_intensity(self, hour: float) -> np.ndarray:
        return self.w.zone_intensity[:, int(hour * 4) % 96] / self._zi_max

    def zone_surge(self, t: float | None = None) -> np.ndarray:
        t = self.t if t is None else t
        out = np.ones(self.n_zones)
        for e in self._surges:
            out[e.zone] += e.at(t)
        return out

    def _advance(self, dt: float) -> None:
        """Move the clock forward, spawning and expiring requests."""
        t1 = self.t + dt
        new = []
        while self._stream_i < len(self._stream) and self._stream[self._stream_i].t_request < t1:
            new.append(self._stream[self._stream_i])
            self._stream_i += 1
        self.stats.seen += len(new)
        self.pending.extend(new)
        if self.record:
            self.stats.log.extend(
                {"ev": "request", "rid": r.rid, "t": r.t_request, "pickup": r.pickup,
                 "dropoff": r.dropoff, "expires": r.expires}
                for r in new
            )
        self.t = t1
        self.pending = [r for r in self.pending if r.expires > self.t]

    def _drive(self, dest: int, kind: str, rid: int | None = None) -> tuple[float, float]:
        """Drive from current node to ``dest``. Returns (seconds, metres)."""
        if dest == self.node:
            return 0.0, 0.0
        hour = self.hour_of(self.t)
        path = self.w.router.path(self.node, dest, hour)
        expected = self.w.router.travel_time(self.node, dest, hour)
        noise = math.exp(self.np_random.normal(0.0, self.w.traffic.trip_noise_sigma))
        secs = expected * noise
        metres = self.w.router.path_length_m(path)
        if self.record:
            self.stats.log.append(
                {"ev": "drive", "kind": kind, "rid": rid, "t0": self.t, "t1": self.t + secs,
                 "path": path}
            )
        self.stats.km_total += metres / 1000
        if kind != "passenger":
            self.stats.km_empty += metres / 1000
        else:
            self.stats.busy_s += secs
        self.stats.cost += COST_PER_KM * metres / 1000
        self._advance(secs)
        self.node = dest
        return secs, metres

    def _leg(self, kind: str, rid: int | None, passenger: bool) -> float:
        """One out-of-network leg between Midtown and LGA. Returns seconds."""
        mult = self.w.city_mult(self.hour_of(self.t))
        noise = math.exp(self.np_random.normal(0.0, 1.5 * self.w.traffic.trip_noise_sigma))
        secs = AIRPORT["leg_freeflow_s"] * mult * noise
        km = AIRPORT["leg_km"]
        if self.record:
            self.stats.log.append({"ev": "offmap", "kind": kind, "rid": rid, "t0": self.t,
                                   "t1": self.t + secs})
        self.stats.km_total += km
        self.stats.cost += COST_PER_KM * km
        if passenger:
            self.stats.busy_s += secs
        else:
            self.stats.km_empty += km
        self._advance(secs)
        return secs

    def _airport_trip(self, req: Request) -> float:
        """Pickup → exit node → LGA → (queue for a return rider | deadhead) → back in Midtown."""
        hour = self.hour_of(self.t)
        s_in, m_in = self._drive(self.w.exit_node, "passenger", req.rid)
        s_out = self._leg("to_airport", req.rid, passenger=True)
        fare = fare_usd(m_in + AIRPORT["leg_km"] * 1000, s_in + s_out, hour) * req.surge
        fare += AIRPORT["surcharge"]
        self.stats.airport_trips += 1
        if self.record:
            self.stats.log.append({"ev": "dropoff", "rid": req.rid, "t": self.t,
                                   "fare": round(fare, 2), "airport": True})
        if self.np_random.random() < AIRPORT["return_fare_prob"]:
            wait = float(self.np_random.uniform(*AIRPORT["queue_wait_s"]))
            if self.record:
                self.stats.log.append({"ev": "offmap", "kind": "airport_queue", "rid": None,
                                       "t0": self.t, "t1": self.t + wait})
            self._advance(wait)
            r_hour = self.hour_of(self.t)
            s_back = self._leg("return_with_rider", None, passenger=True)
            self.node = self.w.exit_node
            dest = int(self.np_random.integers(self.w.graph.n_nodes))
            s_city, m_city = self._drive(dest, "passenger", None)
            back_fare = fare_usd(AIRPORT["leg_km"] * 1000 + m_city, s_back + s_city, r_hour)
            back_fare += AIRPORT["surcharge"]
            if self.record:
                self.stats.log.append({"ev": "dropoff", "rid": None, "t": self.t,
                                       "fare": round(back_fare, 2), "airport_return": True})
            self.stats.served += 1
            fare += back_fare
        else:
            self._leg("return_empty", None, passenger=False)
            self.node = self.w.exit_node
        return fare

    # ------------------------------------------------------------- candidates
    def _compute_candidates(self) -> None:
        hour = self.hour_of(self.t)
        eta_all = self.w.router.times_from(self.node, hour)
        cands = []
        for r in self.pending:
            eta = float(eta_all[r.pickup])
            if self.t + eta > r.expires:  # can't make it in time
                continue
            arr_hour = self.hour_of(self.t + eta)
            dest = self.w.exit_node if r.airport else r.dropoff
            trip_s = self.w.router.travel_time(r.pickup, dest, arr_hour)
            trip_m = self.w.router.path_length_m(self.w.router.path(r.pickup, dest, arr_hour))
            extra = 0.0
            if r.airport:
                trip_s += AIRPORT["leg_freeflow_s"] * self.w.city_mult(arr_hour)
                trip_m += AIRPORT["leg_km"] * 1000
                extra = AIRPORT["surcharge"]
            fare = fare_usd(trip_m, trip_s, arr_hour) * r.surge + extra
            cands.append(Candidate(r, eta, trip_s, trip_m, fare))
        cands.sort(key=lambda c: c.eta_s)
        self.candidates = cands[: self.n_slots]
        self.zone_eta = np.array(
            [eta_all[c] for c in self.w.zones.center_nodes], dtype=np.float64
        )

    def action_masks(self) -> np.ndarray:
        m = np.zeros(self.action_space.n, dtype=bool)
        m[: len(self.candidates)] = True
        cur_zone = self.w.zones.labels[self.node]
        for z in range(self.n_zones):
            m[self.n_slots + z] = z != cur_zone
        m[-1] = True
        return m

    # ------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        hour = self.hour_of(self.t)
        zi_now = self._zone_intensity(hour)
        zi_soon = self._zone_intensity(hour + 0.5)
        cur_zone = self.w.zones.labels[self.node]
        x, y = self.w.xy_norm[self.node]
        g = [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            1.0 - self.t / self.shift_s,
            x,
            y,
            zi_now[cur_zone],
            self.w.city_mult(hour) - 1.0,
        ]
        slots = np.zeros((self.n_slots, 8), dtype=np.float32)
        for k, c in enumerate(self.candidates):
            end_hour = self.hour_of(self.t + c.eta_s + c.trip_s)
            dest = self.w.exit_node if c.req.airport else c.req.dropoff
            dz = self.w.zones.labels[dest]
            slots[k] = [
                1.0,
                c.eta_s / 1200.0,
                c.trip_s / 1800.0,
                c.fare / 40.0,
                (c.req.expires - self.t - c.eta_s) / 600.0,
                self._zone_intensity(end_hour)[dz],
                float(c.req.airport),
                c.req.surge - 1.0,
            ]
        counts = np.bincount(
            [self.w.zones.labels[r.pickup] for r in self.pending], minlength=self.n_zones
        )
        zones = np.stack(
            [self.zone_eta / 1200.0, zi_now, zi_soon, counts / 5.0, self.zone_surge() - 1.0],
            axis=1,
        ).astype(np.float32)
        obs = np.concatenate([np.asarray(g, np.float32), slots.ravel(), zones.ravel()])
        return np.clip(obs, -1.0, 5.0).astype(np.float32)

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        start = options.get("start_hour", self.fixed_start)
        self.start_hour = (
            float(start) if start is not None else float(self.np_random.integers(0, 24))
        )
        self.pending: list[Request] = []
        self.stats = EpisodeStats()
        self.node = int(options.get("start_node", self.np_random.integers(self.w.graph.n_nodes)))
        # Demand is exogenous: the whole shift's rider stream is drawn up front from its own
        # RNG, so different policies on the same seed face exactly the same riders.
        demand_rng = np.random.default_rng(int(self.np_random.integers(2**62)))
        self._stream, self._surges = self.w.demand.sample_shift(
            demand_rng, -300.0, self.shift_s + 7200.0, self.hour_of, self.w.zones.labels,
            self.n_zones,
        )
        if self.record:
            self.stats.log.extend(
                {"ev": "surge", "zone": e.zone, "t": e.t0, "peak": e.peak} for e in self._surges
            )
        self._stream_i = 0
        # Warm-up: start the clock 5 minutes early so some demand is already open.
        self.t = -300.0
        self._advance(300.0)
        self._compute_candidates()
        return self._obs(), {"start_hour": self.start_hour, "start_node": self.node}

    def step(self, action: int):
        action = int(action)
        mask = self.action_masks()
        if not mask[action]:
            action = self.action_space.n - 1  # invalid → wait (never hit with masking)
        reward_usd = 0.0
        cost_before = self.stats.cost
        t_before = self.t
        info: dict = {}

        if action < self.n_slots:
            cand = self.candidates[action]
            req = cand.req
            self.pending = [r for r in self.pending if r.rid != req.rid]
            self._drive(req.pickup, "to_pickup", req.rid)
            if self.t > req.expires:
                self.stats.cancelled += 1
                reward_usd -= CANCEL_PENALTY
                info["event"] = "cancelled"
                if self.record:
                    self.stats.log.append({"ev": "cancel", "rid": req.rid, "t": self.t})
            else:
                if self.record:
                    self.stats.log.append({"ev": "pickup", "rid": req.rid, "t": self.t})
                if req.airport:
                    fare = self._airport_trip(req)
                    info["event"] = "airport"
                else:
                    hour = self.hour_of(self.t)
                    secs, metres = self._drive(req.dropoff, "passenger", req.rid)
                    fare = fare_usd(metres, secs, hour) * req.surge
                    info["event"] = "served"
                    if self.record:
                        self.stats.log.append(
                            {"ev": "dropoff", "rid": req.rid, "t": self.t, "fare": round(fare, 2)}
                        )
                self.stats.revenue += fare
                self.stats.served += 1
                self.stats.surge_trips += int(req.surge > 1.05)
                reward_usd += fare
                info["fare"] = fare
        elif action < self.n_slots + self.n_zones:
            z = action - self.n_slots
            self._drive(int(self.w.zones.center_nodes[z]), "reposition")
            info["event"] = f"reposition:{z}"
        else:
            if self.record:
                self.stats.log.append(
                    {"ev": "wait", "t0": self.t, "t1": self.t + WAIT_S, "node": self.node}
                )
            self._advance(WAIT_S)
            info["event"] = "wait"

        reward_usd -= self.stats.cost - cost_before
        reward_usd -= self.opp_cost * (self.t - t_before) / 3600.0
        terminated = self.t >= self.shift_s
        if not terminated:
            self._compute_candidates()
        else:
            self.candidates = []
            info["episode_stats"] = self.stats.summary(max(self.t, self.shift_s))
        return self._obs(), reward_usd / REWARD_SCALE, terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            return (
                f"t={self.t / 60:6.1f}min hour={self.hour_of(self.t):5.2f} node={self.node} "
                f"pending={len(self.pending)} served={self.stats.served} "
                f"profit=${self.stats.revenue - self.stats.cost:7.2f}"
            )
        return None
