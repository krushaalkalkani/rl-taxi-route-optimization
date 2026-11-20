"""Stage 1 — tabular Q-learning on Gymnasium's classic ``Taxi-v4``.

The 5×5 grid world from the Gymnasium docs: 500 discrete states, 6 actions.
It's a warm-up that grounds the concepts (states, actions, reward shaping,
exploration) before scaling to the real-city environment.

    uv run python -m taxi_rl.taxi_grid
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def train(episodes: int = 5000, alpha: float = 0.1, gamma: float = 0.99, seed: int = 0):
    env = gym.make("Taxi-v4")
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    eps, eps_min, decay = 1.0, 0.02, 0.999
    curve = []
    window: list[float] = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            a = int(rng.integers(6)) if rng.random() < eps else int(np.argmax(q[s]))
            s2, r, term, trunc, _ = env.step(a)
            target = r + (0.0 if term else gamma * q[s2].max())
            q[s, a] += alpha * (target - q[s, a])
            s, total, done = s2, total + r, term or trunc
        eps = max(eps_min, eps * decay)
        window.append(total)
        if (ep + 1) % 50 == 0:
            curve.append({"episode": ep + 1, "mean_return": round(float(np.mean(window)), 2),
                          "epsilon": round(eps, 3)})
            window = []
    return q, curve


def evaluate(q: np.ndarray | None, episodes: int = 200, seed: int = 10_000) -> dict:
    env = gym.make("Taxi-v4")
    rng = np.random.default_rng(seed)
    rets, steps, success = [], [], 0
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, total, n = False, 0.0, 0
        while not done:
            a = int(rng.integers(6)) if q is None else int(np.argmax(q[s]))
            s, r, term, trunc, _ = env.step(a)
            total += r
            n += 1
            done = term or trunc
            success += int(term)
        rets.append(total)
        steps.append(n)
    return {"mean_return": round(float(np.mean(rets)), 2), "mean_steps": round(float(np.mean(steps)), 1),
            "success_rate": round(success / episodes, 3)}


def record_episode(q: np.ndarray, seed: int = 42) -> list[dict]:
    env = gym.make("Taxi-v4")
    s, _ = env.reset(seed=seed)
    frames = []
    done = False
    while not done:
        row, col, pas, dest = env.unwrapped.decode(s)
        a = int(np.argmax(q[s]))
        frames.append({"row": int(row), "col": int(col), "passenger": int(pas), "dest": int(dest), "action": a})
        s, _, term, trunc, _ = env.step(a)
        done = term or trunc
    row, col, pas, dest = env.unwrapped.decode(s)
    frames.append({"row": int(row), "col": int(col), "passenger": int(pas), "dest": int(dest), "action": -1})
    return frames


def main() -> None:
    q, curve = train()
    out = {
        "curve": curve,
        "q_learning": evaluate(q),
        "random": evaluate(None),
        "episode": record_episode(q),
        "hyperparams": {"episodes": 5000, "alpha": 0.1, "gamma": 0.99, "eps_decay": 0.999},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "taxi_v4.json").write_text(json.dumps(out, indent=1))
    print("Q-learning:", out["q_learning"], " random:", out["random"])


if __name__ == "__main__":
    main()
