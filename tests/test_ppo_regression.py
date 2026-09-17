"""PPO regression: saved checkpoints must beat random on held-out scenarios.

Fast (few episodes, CPU) so it runs in CI. Locks in the headline result
for Hyde: ppo_tool test mean_dense >= 0.9.
"""

from pathlib import Path

import finrl.rl  # noqa: F401

CKPT = Path(__file__).parent.parent / "checkpoints" / "ppo_tool.zip"


def test_ppo_checkpoint_exists():
    assert CKPT.exists(), "train first: python -m finrl.rl.train_sb3 --env tool"


def test_ppo_beats_random_on_test():
    import gymnasium as gym
    from stable_baselines3 import PPO

    from finrl.rl.splits import get_split

    model = PPO.load(str(CKPT))
    scenarios = [str(p) for p in get_split("test")][:4]
    assert len(scenarios) == 4
    dense_sum = 0.0
    for i, s in enumerate(scenarios):
        env = gym.make("finrl/Rule605Tool-v0", max_steps=12)
        obs, _ = env.reset(seed=500 + i, options={"scenario": s})
        done = False
        steps = 0
        info: dict = {}
        while not done and steps < 12:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            steps += 1
        dense_sum += float((info.get("dense") or {}).get("reward", 0.0))
        env.close()
    mean_dense = dense_sum / len(scenarios)
    assert mean_dense >= 0.9, f"PPO regressed: mean_dense={mean_dense}"


def test_ppo_learning_curve_artifact():
    import json

    curve_p = Path(__file__).parent.parent / "checkpoints" / "ppo_tool_curve.json"
    assert curve_p.exists()
    curve = json.loads(curve_p.read_text())
    assert len(curve) >= 2
    assert curve[-1]["mean_reward"] >= curve[0]["mean_reward"] - 0.5
