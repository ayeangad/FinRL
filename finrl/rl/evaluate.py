"""Unified RL evaluation harness (classic track).

Evaluates random / REINFORCE-linear / PPO policies on a scenario split and
writes a JSON artifact + console table. Metrics: mean dense reward (the
verifiable report score), success rate (dense >= 0.95), mean episode
length, coverage.

Usage:
  python -m finrl.rl.evaluate --env tool --split test --episodes 2
  python -m finrl.rl.evaluate --env compose --policies random,ppo --ppo-path checkpoints/ppo_compose.zip
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _dense_of_info(info: dict) -> float:
    try:
        return float((info.get("dense") or {}).get("reward", 0.0))
    except Exception:
        return 0.0


def eval_random(env_id: str, scenarios: list[str], max_steps: int, seed: int = 0) -> dict:
    import gymnasium as gym

    import finrl.rl  # noqa: F401

    rng_seed = seed
    dense_list: list[float] = []
    lens: list[int] = []
    per_scenario: list[dict] = []
    for s in scenarios:
        env = gym.make(env_id, max_steps=max_steps)
        obs, _ = env.reset(seed=rng_seed, options={"scenario": s})
        done = False
        steps = 0
        info: dict = {}
        while not done and steps < max_steps:
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            steps += 1
        d = _dense_of_info(info)
        dense_list.append(d)
        lens.append(steps)
        per_scenario.append({"scenario": Path(s).stem, "dense": d, "steps": steps})
        env.close()
        rng_seed += 1
    import numpy as np

    return {
        "policy": "random",
        "n": len(scenarios),
        "mean_dense": round(float(np.mean(dense_list)) if dense_list else 0.0, 4),
        "std_dense": round(float(np.std(dense_list)) if dense_list else 0.0, 4),
        "success_rate": round(float(sum(1 for d in dense_list if d >= 0.95) / max(1, len(dense_list))), 4),
        "mean_steps": round(float(np.mean(lens)) if lens else 0.0, 2),
        "per_scenario": per_scenario,
    }


def eval_reinforce(env_id: str, scenarios: list[str], max_steps: int, checkpoint: str, seed: int = 0) -> dict:
    import gymnasium as gym

    import finrl.rl  # noqa: F401
    from finrl.rl.policy import SoftmaxToolPolicy

    if env_id != "finrl/Rule605Tool-v0":
        return {"policy": "reinforce", "skipped": f"linear policy only supports {env_id}"}
    pol = SoftmaxToolPolicy.load(checkpoint)
    dense_list: list[float] = []
    lens: list[int] = []
    for i, s in enumerate(scenarios):
        env = gym.make(env_id, max_steps=max_steps).unwrapped
        env.reset(s, seed=seed + i)
        # Mirror train.py rollout: SAMPLED tool choice + round-robin slots
        # (greedy collapses to always-classify; sampling matches training).
        order_ids = list(env._order_ids) or ["O1"]
        k = 0
        done = False
        steps = 0
        info: dict = {}
        while not done and steps < max_steps:
            f = env.observation_features()
            a, _ = pol.sample(f)
            oid = order_ids[k % len(order_ids)]
            _, r, term, trunc, info = env.step(a, order_id=oid)
            done = bool(term or trunc)
            if env.legacy_state()["coverage"] >= 1.0 and not done:
                _, _, term2, trunc2, info = env.step(2)
                done = bool(term2 or trunc2)
                steps += 1
                break
            k += 1
            steps += 1
        dense_list.append(_dense_of_info(info))
        lens.append(steps)
        env.close()
    import numpy as np

    return {
        "policy": "reinforce",
        "checkpoint": checkpoint,
        "n": len(scenarios),
        "mean_dense": round(float(np.mean(dense_list)) if dense_list else 0.0, 4),
        "std_dense": round(float(np.std(dense_list)) if dense_list else 0.0, 4),
        "success_rate": round(float(sum(1 for d in dense_list if d >= 0.95) / max(1, len(dense_list))), 4),
        "mean_steps": round(float(np.mean(lens)) if lens else 0.0, 2),
    }


def eval_ppo(env_id: str, scenarios: list[str], max_steps: int, model_path: str) -> dict:
    import gymnasium as gym
    from stable_baselines3 import PPO

    import finrl.rl  # noqa: F401

    model = PPO.load(model_path)
    dense_list: list[float] = []
    lens: list[int] = []
    per_scenario: list[dict] = []
    for i, s in enumerate(scenarios):
        env = gym.make(env_id, max_steps=max_steps)
        obs, _ = env.reset(seed=1000 + i, options={"scenario": s})
        done = False
        steps = 0
        info: dict = {}
        while not done and steps < max_steps:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            steps += 1
        d = _dense_of_info(info)
        dense_list.append(d)
        lens.append(steps)
        per_scenario.append({"scenario": Path(s).stem, "dense": d, "steps": steps})
        env.close()
    import numpy as np

    return {
        "policy": "ppo",
        "model": model_path,
        "n": len(scenarios),
        "mean_dense": round(float(np.mean(dense_list)) if dense_list else 0.0, 4),
        "std_dense": round(float(np.std(dense_list)) if dense_list else 0.0, 4),
        "success_rate": round(float(sum(1 for d in dense_list if d >= 0.95) / max(1, len(dense_list))), 4),
        "mean_steps": round(float(np.mean(lens)) if lens else 0.0, 2),
        "per_scenario": per_scenario,
    }


def run_eval(
    env: str = "tool",
    split: str = "test",
    episodes: int = 1,
    policies: list[str] | None = None,
    ppo_path: str | None = None,
    reinforce_ckpt: str = "checkpoints/tool_policy.npz",
    max_steps: int = 12,
    seed: int = 0,
    out: str | Path = "experiments/rl/eval.json",
) -> dict:
    from finrl.rl.splits import get_split

    env_id = {
        "tool": "finrl/Rule605Tool-v0",
        "compose": "finrl/Rule605Compose-v0",
        "full": "finrl/Rule605ToolFull-v0",
    }[env]
    scenarios = [str(p) for p in get_split(split)] * max(1, episodes)
    policies = policies or ["random"]
    results: dict = {
        "env": env_id,
        "split": split,
        "max_steps": max_steps,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policies": {},
    }
    if "random" in policies:
        results["policies"]["random"] = eval_random(env_id, scenarios, max_steps, seed)
    if "reinforce" in policies:
        try:
            results["policies"]["reinforce"] = eval_reinforce(env_id, scenarios, max_steps, reinforce_ckpt, seed)
        except Exception as e:
            results["policies"]["reinforce"] = {"policy": "reinforce", "error": str(e)}
    if "ppo" in policies:
        mp = ppo_path or f"checkpoints/ppo_{env}.zip"
        try:
            results["policies"]["ppo"] = eval_ppo(env_id, scenarios, max_steps, mp)
        except Exception as e:
            results["policies"]["ppo"] = {"policy": "ppo", "error": str(e)}
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RL policies")
    ap.add_argument("--env", choices=["tool", "compose", "full"], default="tool")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--policies", default="random")
    ap.add_argument("--ppo-path", default=None)
    ap.add_argument("--reinforce-ckpt", default="checkpoints/tool_policy.npz")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/rl/eval.json")
    args = ap.parse_args()
    res = run_eval(
        env=args.env,
        split=args.split,
        episodes=args.episodes,
        policies=[p.strip() for p in args.policies.split(",") if p.strip()],
        ppo_path=args.ppo_path,
        reinforce_ckpt=args.reinforce_ckpt,
        max_steps=args.max_steps,
        seed=args.seed,
        out=args.out,
    )
    # Console table.
    print(f"env={res['env']} split={res['split']}")
    for name, r in res["policies"].items():
        if "error" in r or "skipped" in r:
            print(f"  {name:10s} {r}")
        else:
            print(f"  {name:10s} mean_dense={r['mean_dense']:.4f}±{r['std_dense']:.4f} success={r['success_rate']:.3f} steps={r['mean_steps']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
