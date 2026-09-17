"""30-second Hyde demo: gym.make -> random rollout -> PPO predict -> dense reward.

Usage:
  python -m finrl.rl.demo
  python -m finrl.rl.demo --env compose --scenario scenarios/v0.1/golden/market_01.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def run_demo(env_id: str = "finrl/Rule605Tool-v0", scenario: str = "scenarios/v0.1/golden/market_01.json", max_steps: int = 12, seed: int = 0) -> dict:
    import gymnasium as gym

    import finrl.rl  # noqa: F401

    env = gym.make(env_id, max_steps=max_steps)
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    print(f"env={env_id} scenario={info.get('scenario_id')} n_orders={info.get('n_orders')}")
    print(f"action_space={env.action_space} obs_keys={sorted(obs.keys())}")
    done = False
    steps = 0
    total = 0.0
    while not done and steps < max_steps:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += float(r)
        done = bool(term or trunc)
        steps += 1
        print(f"  step {steps}: action={list(a) if hasattr(a, '__len__') else a} reward={float(r):+.4f} done={done}")
    dense = (info.get("dense") or {}).get("reward", 0.0)
    print(f"random rollout: steps={steps} total_shaped={total:+.4f} dense={float(dense):.4f}")
    # PPO zero-shot predict (untrained init) to prove SB3 wiring.
    try:
        from stable_baselines3 import PPO

        model = PPO("MultiInputPolicy", env, seed=seed, verbose=0)
        obs, _ = env.reset(seed=seed, options={"scenario": scenario})
        a, _ = model.predict(obs, deterministic=True)
        print(f"PPO(MultiInputPolicy) predict OK: {list(a)}")
        ppo_ok = True
    except Exception as e:
        print(f"PPO wiring failed: {e}")
        ppo_ok = False
    env.close()
    return {"env": env_id, "steps": steps, "dense": float(dense), "ppo_ok": ppo_ok}


def main() -> None:
    ap = argparse.ArgumentParser(description="30-sec Rule 605 RL demo")
    ap.add_argument("--env", choices=["tool", "compose", "full"], default="tool")
    ap.add_argument("--scenario", default="scenarios/v0.1/golden/market_01.json")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env_id = {"tool": "finrl/Rule605Tool-v0", "compose": "finrl/Rule605Compose-v0", "full": "finrl/Rule605ToolFull-v0"}[args.env]
    run_demo(env_id, args.scenario, args.max_steps, args.seed)


if __name__ == "__main__":
    main()
