"""Evaluate policies on a fixed set of seeded shifts (common random numbers)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .baselines import POLICIES
from .envs.city_taxi import CityTaxiEnv

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def ppo_policy(model_path: str | Path) -> Callable[[CityTaxiEnv], int]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="cpu")

    def act(env: CityTaxiEnv) -> int:
        a, _ = model.predict(env._obs(), action_masks=env.action_masks(), deterministic=True)
        return int(a)

    return act


def run_policy(
    policy: Callable[[CityTaxiEnv], int],
    seeds: list[int],
    start_hours: list[int] | None = None,
    env: CityTaxiEnv | None = None,
) -> list[dict]:
    env = env or CityTaxiEnv()
    out = []
    for i, s in enumerate(seeds):
        opts = {"start_hour": start_hours[i]} if start_hours else None
        env.reset(seed=s, options=opts)
        done = False
        info = {}
        while not done:
            _, _, done, _, info = env.step(policy(env))
        stats = info["episode_stats"]
        stats["seed"] = s
        stats["start_hour"] = env.start_hour
        out.append(stats)
    return out


def summarise(rows: list[dict]) -> dict:
    keys = ["profit_per_hour", "revenue", "served", "cancelled", "utilization", "empty_ratio", "km_total"]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": round(float(v.mean()), 3), "ci95": round(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ppo_city_taxi.zip")
    ap.add_argument("--episodes", type=int, default=480)
    ap.add_argument("--out", default=str(RESULTS_DIR / "evaluation.json"))
    args = ap.parse_args()

    # 20 shifts per start hour → every hour of the day is covered evenly.
    seeds = [50_000 + i for i in range(args.episodes)]
    hours = [i % 24 for i in range(args.episodes)]
    policies: dict[str, Callable] = {k: v for k, v in POLICIES.items()}
    if Path(args.model).exists():
        policies["ppo"] = ppo_policy(args.model)

    env = CityTaxiEnv()
    results = {}
    per_hour = {}
    raw: dict[str, np.ndarray] = {}
    for name, pol in policies.items():
        rows = run_policy(pol, seeds, hours, env)
        raw[name] = np.array([r["profit_per_hour"] for r in rows])
        results[name] = summarise(rows)
        ph = {}
        for r in rows:
            ph.setdefault(int(r["start_hour"]), []).append(r["profit_per_hour"])
        per_hour[name] = {h: round(float(np.mean(v)), 2) for h, v in sorted(ph.items())}
        print(f"{name:10s} profit/h = {results[name]['profit_per_hour']['mean']:7.2f} "
              f"± {results[name]['profit_per_hour']['ci95']:.2f}  util={results[name]['utilization']['mean']:.3f}")

    # Paired comparison: same seed ⇒ same riders, so per-shift differences cancel most noise.
    paired = {}
    for a in [k for k in raw if k != "nearest"]:
        d = raw[a] - raw["nearest"]
        paired[a] = {
            "mean_diff": round(float(d.mean()), 3),
            "ci95": round(float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), 3),
            "pct": round(float(100 * d.mean() / raw["nearest"].mean()), 2),
            "win_rate": round(float((d > 0).mean()), 3),
        }
        print(f"  {a:10s} vs nearest: {paired[a]['mean_diff']:+.2f} ± {paired[a]['ci95']:.2f} $/h "
              f"({paired[a]['pct']:+.1f}%), better on {paired[a]['win_rate']:.0%} of shifts")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"episodes": args.episodes, "summary": results, "per_start_hour": per_hour,
                    "paired_vs_nearest": paired, "traffic_source": env.w.traffic.source}, indent=2)
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
