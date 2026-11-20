# Midtown Cab RL: taxi route optimization with reinforcement learning

[![CI](https://github.com/krushaalkalkani/rl-taxi-route-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/krushaalkalkani/rl-taxi-route-optimization/actions/workflows/ci.yml)

**Live demo:** <!-- DEMO_URL -->

A reinforcement-learning agent that works a full taxi shift on **Midtown Manhattan's real street
network**. It decides which rider to take, when to turn down a LaGuardia run, and where to wait,
under traffic that changes by the hour and can be calibrated against the **Google Maps Routes API**.

The project goes from Gymnasium's toy `Taxi-v4` grid to a custom Gymnasium environment built on
OpenStreetMap data, trains a `MaskablePPO` agent with Stable-Baselines3, and compares it with
hand-written dispatch rules on identical, seeded shifts.

<!-- RESULTS -->

## How it works

```
OpenStreetMap ──► road graph (OSMnx) ──► hourly traffic model ◄── Google Maps Routes API (calibration)
                                               │
        rider demand (11 real hotspots, surge events, LGA airport runs)
                                               │
                         CityTaxi-v0 (Gymnasium, semi-MDP)
                                               │
                   MaskablePPO (Stable-Baselines3) vs. heuristic baselines
                                               │
                     evaluation (240 seeded shifts) ──► JSON ──► Next.js site on Vercel
```

| Piece | What it does | Where |
|---|---|---|
| Road network | Midtown (14th–57th St, 9th–1st Ave) from OpenStreetMap: 728 intersections, 1,404 directed segments, largest strongly connected component | `src/taxi_rl/graph.py` |
| Traffic + routing | Zone × hour congestion table; all-pairs traffic-aware shortest paths via SciPy Dijkstra, cached per hour, so routes really change with traffic | `src/taxi_rl/traffic.py` |
| Google Maps | `computeRouteMatrix` (TRAFFIC_AWARE) samples per zone and hour → fits congestion + OSM speed correction; `validate` reports simulator-vs-Google ETA error | `src/taxi_rl/gmaps.py` |
| Demand | Non-homogeneous Poisson riders around real hotspots with daily profiles, surge events (higher demand and fares, 20-min half-life), LaGuardia riders with traffic-dependent out-and-back legs | `src/taxi_rl/demand.py` |
| Environment | `CityTaxi-v0`: actions = serve one of the 5 closest reachable riders, reposition to one of 8 zones, or wait; 87-dim observation; action masks; reward = fare − driving cost | `src/taxi_rl/envs/city_taxi.py` |
| Agent | `sb3_contrib.MaskablePPO`, 256×256 actor/critic, γ = 0.995, reward normalisation, 10 parallel envs | `src/taxi_rl/train.py` |
| Baselines | random, nearest rider, best fare-rate, hand-tuned rules (surge chasing, rush-hour airport rule) | `src/taxi_rl/baselines.py` |
| Evaluation | Common random numbers: each shift's riders come from their own RNG stream, so all policies face the same riders. 240 shifts × all start hours, 95% CIs | `src/taxi_rl/evaluate.py` |
| Warm-up | Tabular Q-learning on Gymnasium `Taxi-v4` | `src/taxi_rl/taxi_grid.py` |
| Website | Next.js + Leaflet replay of both cabs side by side, charts with Recharts | `web/` |

### Why this is a real decision problem

With plenty of riders nearby, "take the closest rider" is hard to beat, and the first version of
this environment showed it. Two realistic mechanics give the agent decisions worth learning:

- **Surge zones.** Short-lived demand and price spikes (a show letting out, a storm) are worth
  driving toward, but only if you arrive before they fade.
- **Airport runs.** A LaGuardia rider pays a lot, but the cab leaves Midtown and may come back
  empty. The same trip is great at 2 AM and a trap at 5:30 PM.

## Quickstart

```bash
git clone https://github.com/krushaalkalkani/rl-taxi-route-optimization
cd rl-taxi-route-optimization
uv sync                       # Python 3.11+, installs Gymnasium, SB3, OSMnx, PyTorch
make test                     # offline tests on a synthetic grid
make grid                     # Stage 1: Q-learning on Taxi-v4 (seconds)
make train                    # Stage 2: MaskablePPO, ~30 min on a laptop CPU
make evaluate                 # 240 seeded shifts, all policies → results/evaluation.json
make export                   # JSON for the website
cd web && npm install && npm run dev
```

Use the environment directly:

```python
import gymnasium as gym
import taxi_rl  # registers CityTaxi-v0

env = gym.make("CityTaxi-v0", shift_hours=4).unwrapped
obs, info = env.reset(seed=0, options={"start_hour": 17})
obs, reward, terminated, truncated, info = env.step(env.action_masks().nonzero()[0][0])
```

### Google Maps calibration

```bash
cp .env.example .env          # add a key with the Routes API enabled
make calibrate                # ~576 traffic-aware matrix elements; asks before spending
make validate                 # MAPE of simulator vs Google ETAs at 18:00
make train evaluate export    # retrain on the calibrated traffic
```

The calibration writes `data/traffic_calibration.json`, which `TrafficModel.auto` loads
automatically. Without it, the simulator uses a built-in rush-hour profile.

## Limitations

- Rider demand is synthetic, shaped around real hotspots, not replayed from NYC TLC trip records.
- A single cab. Competition shows up only as riders who give up waiting.
- Traffic varies by zone and hour, not by street or incident.
- The margin over a well-tuned heuristic is modest. That's an honest result, not a bug.

## Ideas for extension

- Replay NYC TLC trip records (zone-level) as the demand stream.
- A multi-agent fleet with shared demand (MARL / centralised dispatch).
- Edge-level routing under stochastic traffic as a second environment.

## Tech

Python · Gymnasium · Stable-Baselines3 / sb3-contrib · PyTorch · OSMnx · NetworkX · SciPy ·
Google Maps Routes API · Next.js · Leaflet · Recharts · Vercel · GitHub Actions
