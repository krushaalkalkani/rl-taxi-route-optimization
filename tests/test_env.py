"""Tests run on the synthetic grid so CI needs no network or API keys."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import taxi_rl  # noqa: F401  (registers CityTaxi-v0)
from taxi_rl.baselines import POLICIES
from taxi_rl.demand import fare_usd
from taxi_rl.envs.city_taxi import CityTaxiEnv
from taxi_rl.graph import synthetic_grid
from taxi_rl.traffic import Router, TrafficModel, Zones


@pytest.fixture(scope="module")
def env():
    return CityTaxiEnv(graph_source="grid", use_calibration=False, shift_hours=1.0)


def test_gymnasium_api_compliance():
    e = gym.make("CityTaxi-v0", graph_source="grid", use_calibration=False, shift_hours=0.5)
    check_env(e.unwrapped, skip_render_check=True)


def test_grid_is_strongly_connected():
    g = synthetic_grid()
    zones = Zones.build(g, 4)
    r = Router(g, zones, TrafficModel.default(zones, g))
    d = np.stack([r.times_from(u, 3.0) for u in range(g.n_nodes)])
    assert np.isfinite(d).all()


def test_rush_hour_is_slower_than_night():
    g = synthetic_grid()
    zones = Zones.build(g, 4)
    r = Router(g, zones, TrafficModel.default(zones, g))
    u, v = 0, g.n_nodes - 1
    assert r.travel_time(u, v, 17.5) > 1.3 * r.travel_time(u, v, 3.0)


def test_path_is_consistent_with_distance():
    g = synthetic_grid()
    zones = Zones.build(g, 4)
    r = Router(g, zones, TrafficModel.default(zones, g))
    p = r.path(3, 40, 9.0)
    assert p[0] == 3 and p[-1] == 40
    t = r.edge_times(9.5)[r.path_edges(p)].sum()
    assert t == pytest.approx(r.travel_time(3, 40, 9.0), rel=1e-4)


def test_masks_and_episode_terminate(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    steps = 0
    done = False
    while not done:
        mask = env.action_masks()
        assert mask[-1]  # wait is always valid
        assert mask[: env.n_slots].sum() == len(env.candidates)
        _, _r, done, _, info = env.step(int(np.flatnonzero(mask)[0]))
        steps += 1
        assert steps < 1000
    assert "episode_stats" in info


def test_same_seed_same_riders(env):
    env.reset(seed=7, options={"start_hour": 8})
    a = [(r.pickup, r.dropoff, r.t_request) for r in env._stream]
    env.reset(seed=7, options={"start_hour": 8})
    b = [(r.pickup, r.dropoff, r.t_request) for r in env._stream]
    assert a == b and len(a) > 0


def test_greedy_beats_random(env):
    def run(policy):
        tot = 0.0
        for s in range(6):
            env.reset(seed=100 + s, options={"start_hour": 18})
            done = False
            while not done:
                _, _, done, _, info = env.step(policy(env))
            tot += info["episode_stats"]["profit"]
        return tot

    assert run(POLICIES["nearest"]) > run(POLICIES["random"])


def test_fare_formula():
    assert fare_usd(1609.34, 600, 3) == pytest.approx(3 + 3.5 + 3.5)
    assert fare_usd(1609.34, 600, 17) == pytest.approx(3 + 3.5 + 3.5 + 1.0)
