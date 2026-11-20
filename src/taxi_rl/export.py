"""Export everything the showcase website needs as static JSON.

    uv run python -m taxi_rl.export --out ../web/public/data
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .baselines import nearest_request
from .envs.city_taxi import CityTaxiEnv
from .evaluate import RESULTS_DIR, ppo_policy


def _r(x: float) -> float:
    return round(float(x), 5)


def network_json(env: CityTaxiEnv) -> dict:
    g = env.w.graph
    z = env.w.zones
    return {
        "name": g.name,
        "n_nodes": g.n_nodes,
        "n_edges": g.n_edges,
        "bounds": [[_r(g.lat.min()), _r(g.lon.min())], [_r(g.lat.max()), _r(g.lon.max())]],
        "edges": [[[_r(a), _r(b)] for a, b in geom] for geom in g.edge_geometry],
        "edge_zone": env.w.router.edge_zone.tolist(),
        "zones": [
            {"id": i, "lat": _r(g.lat[n]), "lon": _r(g.lon[n])} for i, n in enumerate(z.center_nodes)
        ],
        "hotspots": [{"name": h[0], "lat": h[1], "lon": h[2], "kind": h[3]} for h in env.w.demand.hotspots],
        "congestion": np.round(env.w.traffic.congestion * env.w.traffic.global_scale, 3).tolist(),
        "traffic_source": env.w.traffic.source,
        "demand_by_hour": [
            round(float(env.w.demand.hotspot_rates(h).sum()), 3) for h in range(24)
        ],
    }


def replay(env: CityTaxiEnv, policy, seed: int, start_hour: int, start_node: int) -> dict:
    env.record = True
    env.reset(seed=seed, options={"start_hour": start_hour, "start_node": start_node})
    done = False
    info: dict = {}
    while not done:
        _, _, done, _, info = env.step(policy(env))
    env.record = False
    g = env.w.graph
    router = env.w.router
    drives, waits, reqs, events, offmap, surges = [], [], {}, [], [], []
    for ev in env.stats.log:
        if ev["ev"] == "request":
            reqs[ev["rid"]] = {
                "rid": ev["rid"], "t": round(ev["t"], 1), "expires": round(ev["expires"], 1),
                "pickup": [_r(g.lat[ev["pickup"]]), _r(g.lon[ev["pickup"]])],
                "dropoff": [_r(g.lat[ev["dropoff"]]), _r(g.lon[ev["dropoff"]])],
                "status": "missed",
                "airport": False,
            }
        elif ev["ev"] == "drive":
            path = ev["path"]
            coords: list[list[float]] = []
            cum = [0.0]
            ff = g.edge_freeflow_s[router.path_edges(path)]
            for e_idx in router.path_edges(path):
                geom = g.edge_geometry[e_idx]
                coords.extend([[_r(a), _r(b)] for a, b in (geom if not coords else geom[1:])])
            # time fraction at each *node* of the path (geometry vertices interpolate by index)
            cum = (np.concatenate([[0.0], np.cumsum(ff)]) / max(ff.sum(), 1e-9)).round(4).tolist()
            node_coords = [[_r(g.lat[n]), _r(g.lon[n])] for n in path]
            drives.append({
                "kind": ev["kind"], "rid": ev["rid"], "t0": round(ev["t0"], 1), "t1": round(ev["t1"], 1),
                "nodes": node_coords, "frac": cum, "geom": coords,
            })
        elif ev["ev"] == "offmap":
            offmap.append({"kind": ev["kind"], "t0": round(ev["t0"], 1), "t1": round(ev["t1"], 1)})
        elif ev["ev"] == "surge":
            zc = env.w.zones.center_nodes[ev["zone"]]
            surges.append({"zone": ev["zone"], "t": round(ev["t"], 1), "peak": round(ev["peak"], 2),
                           "at": [_r(g.lat[zc]), _r(g.lon[zc])]})
        elif ev["ev"] == "wait":
            waits.append({"t0": round(ev["t0"], 1), "t1": round(ev["t1"], 1),
                          "at": [_r(g.lat[ev["node"]]), _r(g.lon[ev["node"]])]})
        elif ev["ev"] in ("pickup", "dropoff", "cancel"):
            events.append({k: (round(v, 1) if isinstance(v, float) else v) for k, v in ev.items()})
            if ev.get("rid") is not None and ev["rid"] in reqs:
                reqs[ev["rid"]]["status"] = {"pickup": "picked", "dropoff": "served", "cancel": "cancelled"}[ev["ev"]]
                if ev["ev"] == "pickup":
                    reqs[ev["rid"]]["t_pickup"] = round(ev["t"], 1)
                if ev["ev"] == "dropoff":
                    reqs[ev["rid"]]["t_dropoff"] = round(ev["t"], 1)
                    reqs[ev["rid"]]["fare"] = ev["fare"]
    airport_ids = {r.rid for r in env._stream if r.airport}
    for rid, r in reqs.items():
        r["airport"] = rid in airport_ids
    return {
        "seed": seed,
        "start_hour": start_hour,
        "shift_s": env.shift_s,
        "end_t": round(env.t, 1),
        "stats": info["episode_stats"],
        "drives": drives,
        "waits": waits,
        "offmap": offmap,
        "surges": [x for x in surges if -1800 < x["t"] < env.t],
        "requests": [r for r in reqs.values() if r["t"] <= env.t],
        "events": events,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/data")
    ap.add_argument("--model", default="models/ppo_city_taxi.zip")
    ap.add_argument("--shift-hours", type=float, default=2.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = CityTaxiEnv(shift_hours=args.shift_hours)
    (out / "network.json").write_text(json.dumps(network_json(env), separators=(",", ":")))

    ppo = ppo_policy(args.model)
    scenarios = [("pm-rush", 17, 7), ("am-rush", 7, 11), ("night", 22, 3)]
    for name, hour, seed in scenarios:
        start_node = int(np.random.default_rng(seed).integers(env.w.graph.n_nodes))
        payload = {
            "scenario": name,
            "ppo": replay(env, ppo, seed, hour, start_node),
            "nearest": replay(env, nearest_request, seed, hour, start_node),
        }
        (out / f"replay-{name}.json").write_text(json.dumps(payload, separators=(",", ":")))
        print(name, "ppo", payload["ppo"]["stats"]["profit"], "nearest", payload["nearest"]["stats"]["profit"])

    import gymnasium  # noqa: F401

    from .demand import AIRPORT

    meta = {"airport": AIRPORT, "graph": "OpenStreetMap — Midtown Manhattan"}
    (out / "meta.json").write_text(json.dumps(meta))
    for f in ["evaluation.json", "learning_curve.json", "taxi_v4.json", "gmaps_validation.json"]:
        if (RESULTS_DIR / f).exists():
            shutil.copy(RESULTS_DIR / f, out / f)
    print("exported to", out)


if __name__ == "__main__":
    main()
