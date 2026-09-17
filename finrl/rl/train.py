"""REINFORCE training loop over golden scenarios.

Produces an owned checkpoint artifact (npz) + learning-curve JSON, i.e. the
post-training story Hyde expects: weights learned from verifiable reward,
eval-gated release, operator lineage.

Usage:
    python -m finrl.rl.train --episodes 200 --lr 0.05 --out checkpoints/tool_policy.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from finrl.rl.gym_env import MASKED_TOOLS, Rule605GymEnv
from finrl.rl.policy import SoftmaxToolPolicy

MASKED_SUBMIT = MASKED_TOOLS.index("submit_report")


def rollout(policy: SoftmaxToolPolicy, scenario_path: Path, max_steps: int = 12, greedy: bool = False):
    env = Rule605GymEnv(max_steps=max_steps)
    env.reset(scenario_path)
    feats, actions, rewards = [], [], []
    # Round-robin order binding so multi-order scenarios get coverage.
    order_ids = list(env._order_ids) or ["O1"]
    k = 0
    done = False
    info: dict = {}
    steps = 0
    while not done and steps < max_steps:
        f = env.observation_features()
        if greedy:
            a = policy.greedy(f)
            oid = order_ids[k % len(order_ids)]
            out = env.step(a, order_id=oid)
        else:
            a, _ = policy.sample(f)
            oid = order_ids[k % len(order_ids)]
            out = env.step(a, order_id=oid)
        if len(out) == 5:
            _, r, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            _, r, done, info = out
        feats.append(f)
        actions.append(a)
        rewards.append(float(r))
        # Heuristic: force submit once fully covered (keeps episodes short);
        # the LEARNED part is the tool ordering that reaches coverage fast.
        cov = float(env.legacy_state()["coverage"])
        if cov >= 1.0 and not done:
            out2 = env.step(MASKED_SUBMIT)
            if len(out2) == 5:
                _, r2, terminated, truncated, info = out2
                done = bool(terminated or truncated)
            else:
                _, r2, done, info = out2
            feats.append(env.observation_features())
            actions.append(MASKED_SUBMIT)
            rewards.append(float(r2))
            steps += 1
            break
        k += 1
        steps += 1
    dense_reward = float((info.get("dense") or {}).get("reward", 0.0)) if isinstance(info, dict) else 0.0
    return {"features": feats, "actions": actions, "rewards": rewards, "dense": dense_reward}


def train_reinforce(
    scenarios_dir: str | Path = "scenarios/v0.1/golden",
    episodes: int = 200,
    lr: float = 0.05,
    batch: int = 8,
    max_steps: int = 12,
    seed: int = 0,
    out: str | Path = "checkpoints/tool_policy.npz",
) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    files = sorted(Path(scenarios_dir).glob("*.json"))
    if not files:
        raise ValueError(f"No scenarios in {scenarios_dir}")
    policy = SoftmaxToolPolicy(seed=seed)
    history = []
    pending = []
    for ep in range(1, episodes + 1):
        path = Path(rng.choice([str(f) for f in files]))
        traj = rollout(policy, path, max_steps=max_steps)
        pending.append(traj)
        if len(pending) >= batch or ep == episodes:
            stats = policy.reinforce_update(pending, lr=lr)
            stats["episode"] = ep
            stats["mean_dense"] = round(float(sum(t["dense"] for t in pending) / len(pending)), 4)
            history.append(stats)
            pending = []
    out_path = policy.save(out)
    curve_path = Path(str(out).replace(".npz", "_curve.json"))
    curve_path.write_text(json.dumps(history, indent=2))
    return {
        "checkpoint": str(out_path),
        "curve": str(curve_path),
        "final_mean_return": history[-1]["mean_return"] if history else 0.0,
        "final_mean_dense": history[-1]["mean_dense"] if history else 0.0,
        "episodes": episodes,
    }


def main():
    p = argparse.ArgumentParser(description="REINFORCE tool-use policy training")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenarios-dir", default="scenarios/v0.1/golden")
    p.add_argument("--out", default="checkpoints/tool_policy.npz")
    args = p.parse_args()
    t0 = time.perf_counter()
    res = train_reinforce(
        scenarios_dir=args.scenarios_dir,
        episodes=args.episodes,
        lr=args.lr,
        batch=args.batch,
        max_steps=args.max_steps,
        seed=args.seed,
        out=args.out,
    )
    res["elapsed_s"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
