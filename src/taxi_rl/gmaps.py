"""Google Maps integration — calibrate and validate the simulator against real traffic.

Uses the **Routes API** ``computeRouteMatrix`` endpoint with
``routingPreference=TRAFFIC_AWARE``. For every (zone, hour) we sample origin /
destination intersections from the road graph, ask Google for the
traffic-aware ``duration`` and the no-traffic ``staticDuration``, and fit:

* ``global_scale``      = median(staticDuration / simulator free-flow time)
                          — corrects OSM speed-limit assumptions
* ``congestion[z, h]``  = median(duration / staticDuration) for zone z at hour h

The result is written to ``data/traffic_calibration.json`` and picked up
automatically by :class:`taxi_rl.traffic.TrafficModel`.

    export GOOGLE_MAPS_API_KEY=...
    uv run python -m taxi_rl.gmaps calibrate --pairs-per-zone 3 --hours 0 6 8 12 17 18 22
    uv run python -m taxi_rl.gmaps validate --n 50

Cost note: each origin×destination element is billed as a Routes "Advanced"
element (traffic-aware). Defaults use ~8 zones × 9 elements × 8 hours ≈ 576
elements, inside the monthly free allowance at the time of writing. The CLI
prints the element count and asks before spending.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from .graph import RoadGraph, load_graph
from .traffic import CALIBRATION_PATH, Router, TrafficModel, Zones

ROUTE_MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
FIELD_MASK = "originIndex,destinationIndex,duration,staticDuration,distanceMeters,condition"
NY_TZ = dt.timezone(dt.timedelta(hours=-5))  # EST; good enough for picking hours


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise SystemExit(
            "GOOGLE_MAPS_API_KEY is not set. Create a key with the Routes API enabled "
            "(https://console.cloud.google.com/google/maps-apis) and put it in .env"
        )
    return key


def _secs(s: str | None) -> float | None:
    return float(s.rstrip("s")) if s else None


def route_matrix(
    origins: list[tuple[float, float]],
    destinations: list[tuple[float, float]],
    departure: dt.datetime,
    key: str,
) -> list[dict]:
    """Call computeRouteMatrix. Returns list of {o, d, duration, static, meters}."""

    def wp(lat, lon):
        return {"waypoint": {"location": {"latLng": {"latitude": lat, "longitude": lon}}}}

    body = {
        "origins": [wp(*o) for o in origins],
        "destinations": [wp(*d) for d in destinations],
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "departureTime": departure.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    headers = {"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELD_MASK, "Content-Type": "application/json"}
    for attempt in range(4):
        resp = requests.post(ROUTE_MATRIX_URL, json=body, headers=headers, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2**attempt)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Routes API {resp.status_code}: {resp.text[:500]}")
        break
    else:
        raise RuntimeError("Routes API kept failing")
    out = []
    for el in resp.json():
        if el.get("condition") != "ROUTE_EXISTS":
            continue
        out.append(
            {
                "o": el.get("originIndex", 0),
                "d": el.get("destinationIndex", 0),
                "duration": _secs(el.get("duration")),
                "static": _secs(el.get("staticDuration")),
                "meters": el.get("distanceMeters"),
            }
        )
    return out


def next_weekday_at(hour: int, now: dt.datetime | None = None) -> dt.datetime:
    """Next Tue–Thu (typical traffic) at ``hour`` NY time, always in the future."""
    now = now or dt.datetime.now(NY_TZ)
    d = now + dt.timedelta(days=1)
    while d.weekday() not in (1, 2, 3):
        d += dt.timedelta(days=1)
    return d.replace(hour=hour, minute=0, second=0, microsecond=0)


def calibrate(
    graph: RoadGraph,
    zones: Zones,
    hours: list[int],
    pairs_per_zone: int = 3,
    seed: int = 0,
    yes: bool = False,
) -> TrafficModel:
    key = api_key()
    rng = np.random.default_rng(seed)
    n_el = zones.n * pairs_per_zone * pairs_per_zone * len(hours)
    print(f"Will request {n_el} traffic-aware matrix elements ({zones.n} zones × "
          f"{pairs_per_zone}² × {len(hours)} hours).")
    if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted")

    base = TrafficModel.default(zones, graph)
    base.global_scale = 1.0
    base.congestion[:] = 1.0
    free_router = Router(graph, zones, base)

    ratios = np.full((zones.n, 24), np.nan)
    scale_samples: list[float] = []
    raw: list[dict] = []
    for h in hours:
        dep = next_weekday_at(h)
        for z in range(zones.n):
            nodes = np.flatnonzero(zones.labels == z)
            o_nodes = rng.choice(nodes, size=pairs_per_zone, replace=False)
            d_nodes = rng.choice(nodes, size=pairs_per_zone, replace=False)
            els = route_matrix(
                [(graph.lat[n], graph.lon[n]) for n in o_nodes],
                [(graph.lat[n], graph.lon[n]) for n in d_nodes],
                dep,
                key,
            )
            zr = []
            for e in els:
                o, d = int(o_nodes[e["o"]]), int(d_nodes[e["d"]])
                if o == d or not e["duration"] or not e["static"]:
                    continue
                ff = free_router.travel_time(o, d, 3.0)
                zr.append(e["duration"] / e["static"])
                if ff > 30:
                    scale_samples.append(e["static"] / ff)
                raw.append({"hour": h, "zone": z, "o": o, "d": d, **e, "sim_freeflow": ff})
            if zr:
                ratios[z, h] = float(np.median(zr))
            print(f"  hour {h:02d} zone {z}: congestion x{ratios[z, h]:.2f}")

    # Fill unmeasured hours by circular interpolation per zone.
    filled = np.ones_like(ratios)
    hrs = np.arange(24)
    for z in range(zones.n):
        ok = ~np.isnan(ratios[z])
        if ok.sum() == 0:
            filled[z] = 1.0
            continue
        xs = hrs[ok]
        ys = ratios[z, ok]
        filled[z] = np.interp(hrs, np.concatenate([xs - 24, xs, xs + 24]), np.tile(ys, 3))
    scale = float(np.median(scale_samples)) if scale_samples else 1.0

    tm = TrafficModel(
        congestion=filled,
        global_scale=scale,
        source="google-routes-api",
        meta={
            "calibrated_at": dt.datetime.now(dt.UTC).isoformat(),
            "hours_measured": hours,
            "elements": len(raw),
            "global_scale": scale,
        },
    )
    tm.save(CALIBRATION_PATH)
    (CALIBRATION_PATH.parent / "gmaps_samples.json").write_text(json.dumps(raw, indent=1))
    print(f"Saved {CALIBRATION_PATH} (global free-flow scale {scale:.3f})")
    return tm


def validate(graph: RoadGraph, zones: Zones, n: int = 40, hour: int = 18, seed: int = 1) -> dict:
    """Compare simulator ETAs with Google's traffic-aware ETAs on random trips."""
    key = api_key()
    tm = TrafficModel.auto(zones, graph)
    router = Router(graph, zones, tm)
    rng = np.random.default_rng(seed)
    pairs = [(int(a), int(b)) for a, b in rng.integers(graph.n_nodes, size=(n, 2)) if a != b]
    dep = next_weekday_at(hour)
    rows = []
    for o, d in pairs:  # 1x1 matrices keep the pairing exact
        els = route_matrix([(graph.lat[o], graph.lon[o])], [(graph.lat[d], graph.lon[d])], dep, key)
        if not els or not els[0]["duration"]:
            continue
        sim = router.travel_time(o, d, hour + 0.5)
        rows.append({"o": o, "d": d, "google_s": els[0]["duration"], "sim_s": sim})
    g = np.array([r["google_s"] for r in rows])
    s = np.array([r["sim_s"] for r in rows])
    report = {
        "hour": hour,
        "n": len(rows),
        "traffic_source": tm.source,
        "mape": round(float(np.mean(np.abs(s - g) / g)), 4),
        "median_ape": round(float(np.median(np.abs(s - g) / g)), 4),
        "pearson_r": round(float(np.corrcoef(g, s)[0, 1]), 4),
        "pairs": rows,
    }
    out = Path(__file__).resolve().parents[2] / "results" / "gmaps_validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"MAPE {report['mape']:.1%}  r={report['pearson_r']:.3f}  ({len(rows)} trips) → {out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--hours", type=int, nargs="+", default=[0, 3, 7, 8, 12, 17, 18, 22])
    c.add_argument("--pairs-per-zone", type=int, default=3)
    c.add_argument("--yes", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--n", type=int, default=40)
    v.add_argument("--hour", type=int, default=18)
    args = ap.parse_args()

    graph = load_graph("osm")
    zones = Zones.build(graph, 8)
    if args.cmd == "calibrate":
        calibrate(graph, zones, args.hours, args.pairs_per_zone, yes=args.yes)
    else:
        validate(graph, zones, args.n, args.hour)


if __name__ == "__main__":
    main()
