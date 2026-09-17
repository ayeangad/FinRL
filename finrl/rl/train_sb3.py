"""SB3 PPO training for Rule 605 envs (classic RL track).

Randomizes the training scenario every episode (hash-stable train split
by default) so the policy generalizes instead of memorizing one JSON.
Eval runs on held-out val/test scenarios.

Usage:
  python -m finrl.rl.train_sb3 --env tool --timesteps 50000 --seed 0
  python -m finrl.rl.train_sb3 --env compose --timesteps 100000 --n-envs 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _make_env_fn(env_id: str, scenarios: list[str], max_steps: int, seed: int, cycle: bool = True):
    import gymnasium as gym

    import finrl.rl  # noqa: F401  (registers envs)

    idx = {"i": 0}

    def _fn():
        env = gym.make(env_id, max_steps=max_steps)
        base_reset = env.reset

        def _reset(*args, **kwargs):
            # Sample scenario per episode (deterministic cycle for eval,
            # seeded random for train when cycle=False... here always cycle
            # through shuffled order for reproducibility).
            if scenarios:
                s = scenarios[idx["i"] % len(scenarios)]
                idx["i"] += 1
                kwargs = dict(kwargs)
                kwargs.setdefault("options", {}).update({"scenario": s})
            return base_reset(*args, **kwargs)

        env.reset = _reset  # type: ignore
        return env

    return _fn


def train_ppo(
    env_id: str = "finrl/Rule605Tool-v0",
    scenarios: list[str] | None = None,
    eval_scenarios: list[str] | None = None,
    timesteps: int = 50000,
    seed: int = 0,
    n_envs: int = 4,
    max_steps: int = 12,
    lr: float = 3e-4,
    out: str | Path = "checkpoints/ppo_tool.zip",
    eval_freq: int = 10000,
) -> dict:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    import finrl.rl  # noqa: F401

    if scenarios is None:
        from finrl.rl.splits import get_split

        try:
            scenarios = [str(p) for p in get_split("train")]
        except Exception:
            scenarios = sorted(str(p) for p in Path("scenarios/v0.1/golden").glob("*.json"))
    if eval_scenarios is None:
        from finrl.rl.splits import get_split

        try:
            eval_scenarios = [str(p) for p in get_split("val")]
        except Exception:
            eval_scenarios = scenarios[:5]

    rng = np.random.default_rng(seed)
    order = list(scenarios)
    rng.shuffle(order)

    def _train_fn():
        import gymnasium as _gym

        env = _gym.make(env_id, max_steps=max_steps)
        # Per-episode random scenario from shuffled train list.
        state = {"i": int(rng.integers(0, len(order)))}
        base_reset = env.reset

        def _reset(*args, **kwargs):
            s = order[state["i"] % len(order)]
            state["i"] += 1
            kw = dict(kwargs)
            opts = dict(kw.get("options") or {})
            opts["scenario"] = s
            kw["options"] = opts
            return base_reset(*args, **kw)

        env.reset = _reset  # type: ignore
        env = Monitor(env)
        return env

    vec = make_vec_env(_train_fn, n_envs=n_envs, seed=seed)

    def _eval_fn():
        import gymnasium as _gym

        env = _gym.make(env_id, max_steps=max_steps)
        state = {"i": 0}
        base_reset = env.reset

        def _reset(*args, **kwargs):
            s = eval_scenarios[state["i"] % len(eval_scenarios)]
            state["i"] += 1
            kw = dict(kwargs)
            opts = dict(kw.get("options") or {})
            opts["scenario"] = s
            kw["options"] = opts
            return base_reset(*args, **kw)

        env.reset = _reset  # type: ignore
        env = Monitor(env)
        return env

    from stable_baselines3.common.env_util import make_vec_env as _mve

    eval_vec = _mve(_eval_fn, n_envs=1, seed=seed + 999)
    eval_dir = Path(str(out)).parent / "sb3_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=max(1000, eval_freq // max(1, n_envs)),
        n_eval_episodes=min(10, max(2, len(eval_scenarios))),
        deterministic=True,
    )
    t0 = time.perf_counter()
    model = PPO("MultiInputPolicy", vec, learning_rate=lr, seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    elapsed = time.perf_counter() - t0
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_p))
    # Learning-curve artifact from SB3 evaluations.
    curve = []
    try:
        ev = np.load(eval_dir / "evaluations.npz")
        for i, (ts, res, lens) in enumerate(zip(ev["timesteps"], ev["results"], ev["ep_lengths"])):
            curve.append(
                {
                    "eval_idx": i,
                    "timesteps": int(ts),
                    "mean_reward": round(float(np.mean(res)), 4),
                    "std_reward": round(float(np.std(res)), 4),
                    "mean_len": round(float(np.mean(lens)), 2),
                }
            )
    except Exception:
        pass
    curve_path = out_p.with_name(out_p.stem + "_curve.json")
    curve_path.write_text(json.dumps(curve, indent=2))
    vec.close()
    eval_vec.close()
    return {
        "model": str(out_p),
        "curve": str(curve_path),
        "timesteps": timesteps,
        "elapsed_s": round(elapsed, 2),
        "env": env_id,
        "n_train": len(order),
        "n_eval": len(eval_scenarios),
        "final_mean_reward": curve[-1]["mean_reward"] if curve else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SB3 PPO training for Rule 605")
    ap.add_argument("--env", choices=["tool", "compose", "full"], default="tool")
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenarios-dir", default="scenarios/v0.1/golden")
    args = ap.parse_args()
    env_id = {"tool": "finrl/Rule605Tool-v0", "compose": "finrl/Rule605Compose-v0", "full": "finrl/Rule605ToolFull-v0"}[args.env]
    out = args.out or f"checkpoints/ppo_{args.env}.zip"
    res = train_ppo(env_id=env_id, timesteps=args.timesteps, seed=args.seed, n_envs=args.n_envs, max_steps=args.max_steps, lr=args.lr, out=out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
