"""Train a MaskablePPO dispatch agent on CityTaxi-v0.

    uv run python -m taxi_rl.train --timesteps 1_000_000 --n-envs 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .baselines import nearest_request
from .envs.city_taxi import CityTaxiEnv
from .evaluate import RESULTS_DIR, run_policy
from .policy import ScoringPolicy

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def make_env(rank: int, opp_cost: float = 0.0):
    def _init():
        env = CityTaxiEnv(opportunity_cost_per_hour=opp_cost)
        env.reset(seed=rank)
        return env

    return _init


class LearningCurve(BaseCallback):
    """Periodically evaluates the greedy (deterministic) policy on held-out shifts."""

    def __init__(self, every: int, seeds: list[int], hours: list[int], baseline: float, name: str):
        super().__init__()
        self.name = name
        self.every = every
        self.seeds = seeds
        self.hours = hours
        self.baseline = baseline
        self.curve: list[dict] = []
        self._next = 0
        self.env = CityTaxiEnv()
        self.best = -np.inf

    def _policy(self, env: CityTaxiEnv) -> int:
        a, _ = self.model.predict(env._obs(), action_masks=env.action_masks(), deterministic=True)
        return int(a)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            rows = run_policy(self._policy, self.seeds, self.hours, self.env)
            pph = float(np.mean([r["profit_per_hour"] for r in rows]))
            self.curve.append({"timesteps": int(self.num_timesteps), "profit_per_hour": round(pph, 2)})
            print(f"[eval] {self.num_timesteps:>9,d} steps  profit/h = {pph:6.2f}  "
                  f"(nearest baseline {self.baseline:.2f})", flush=True)
            if pph > self.best:
                self.best = pph
                self.model.save(MODELS_DIR / self.name)
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opp-cost", type=float, default=68.0,
                    help="reference $/hour charged per unit of time (average-reward shaping)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--name", default="ppo_city_taxi")
    ap.add_argument("--arch", choices=["scoring", "mlp"], default="scoring")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    args = ap.parse_args()

    torch.set_num_threads(2)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    eval_seeds = [90_000 + i for i in range(96)]
    eval_hours = [i % 24 for i in range(96)]
    base_rows = run_policy(nearest_request, eval_seeds, eval_hours)
    baseline = float(np.mean([r["profit_per_hour"] for r in base_rows]))

    venv = SubprocVecEnv([make_env(args.seed * 100 + i, args.opp_cost) for i in range(args.n_envs)])
    # Airport runs make returns heavy-tailed; normalising rewards stabilises the critic.
    # Observations are already scaled in the env, so the saved policy needs no wrapper.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=args.gamma)
    if args.arch == "scoring":
        policy, policy_kwargs = ScoringPolicy, {}
    else:
        policy, policy_kwargs = "MlpPolicy", {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
    model = MaskablePPO(
        policy,
        venv,
        learning_rate=args.lr,
        n_steps=1024,
        batch_size=2048,
        n_epochs=6,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=0,
        device="cpu",
    )
    cb = LearningCurve(args.eval_every, eval_seeds, eval_hours, baseline, args.name)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    cb._next = 0
    cb._on_step()  # final eval
    elapsed = time.time() - t0
    venv.close()

    (RESULTS_DIR / f"learning_curve_{args.name}.json").write_text(
        json.dumps(
            {"curve": cb.curve, "nearest_baseline": round(baseline, 2), "best": round(cb.best, 2),
             "timesteps": args.timesteps, "n_envs": args.n_envs, "train_seconds": round(elapsed),
             "opp_cost": args.opp_cost, "gamma": args.gamma, "arch": args.arch, "lr": args.lr,
             "ent_coef": args.ent},
            indent=2,
        )
    )
    print(f"done in {elapsed / 60:.1f} min — best profit/h {cb.best:.2f} vs nearest {baseline:.2f}")


if __name__ == "__main__":
    main()
