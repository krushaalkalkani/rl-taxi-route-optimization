"""Hand-written dispatch policies used as baselines.

Each policy is a callable ``policy(env) -> action`` that reads the unwrapped
:class:`CityTaxiEnv` (candidates, zone ETAs, masks).
"""

from __future__ import annotations

import numpy as np

from .envs.city_taxi import CityTaxiEnv


def _wait(env: CityTaxiEnv) -> int:
    return env.action_space.n - 1


_RNG = np.random.default_rng(0)


def random_policy(env: CityTaxiEnv) -> int:
    return int(_RNG.choice(np.flatnonzero(env.action_masks())))


def nearest_request(env: CityTaxiEnv) -> int:
    """What most simple dispatchers do: take the closest open request, else wait."""
    if env.candidates:
        return 0  # candidates are sorted by pickup ETA
    return _wait(env)


def best_fare_rate(env: CityTaxiEnv) -> int:
    """Pick the request maximising fare per minute of total time (pickup + trip);
    if nothing is open, drive to the zone with the highest demand-per-minute-of-driving."""
    if env.candidates:
        rates = [c.fare / ((c.eta_s + c.trip_s) / 60.0 + 1.0) for c in env.candidates]
        return int(np.argmax(rates))
    return _reposition_to_demand(env)


def _reposition_to_demand(env: CityTaxiEnv) -> int:
    hour = env.hour_of(env.t)
    zi = env._zone_intensity(hour)
    cur = env.w.zones.labels[env.node]
    if zi[cur] >= 0.8 * zi.max():
        return _wait(env)
    score = zi / (env.zone_eta / 60.0 + 3.0)
    score[cur] = -np.inf
    return env.n_slots + int(np.argmax(score))


def smart_heuristic(env: CityTaxiEnv) -> int:
    """Hand-tuned expert rules: fare-per-minute scoring with surge, decline airport
    runs when the city is congested, and chase surging zones when idle."""
    hour = env.hour_of(env.t)
    congested = env.w.city_mult(hour) > 1.25
    best, best_score = None, -np.inf
    for k, c in enumerate(env.candidates):
        if c.req.airport and congested:
            continue
        total_min = (c.eta_s + c.trip_s) / 60.0 + (20.0 if c.req.airport else 0.0)
        score = c.fare / (total_min + 2.0)
        if score > best_score:
            best, best_score = k, score
    if best is not None:
        return best
    surge = env.zone_surge()
    cur = env.w.zones.labels[env.node]
    z = int(np.argmax(surge))
    if surge[z] > 1.3 and z != cur and env.zone_eta[z] < 600:
        return env.n_slots + z
    return _reposition_to_demand(env)


POLICIES = {
    "random": random_policy,
    "nearest": nearest_request,
    "fare_rate": best_fare_rate,
    "smart": smart_heuristic,
}
