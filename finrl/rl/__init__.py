"""RL package: turns the Rule 605 harness into a trainable MDP.

Components:
  reward.py  - dense, per-column shaped reward (verifiable reward for RLVR)
  gym_env.py - Gymnasium tool-use MDP (MultiDiscrete, evidence-conditioned submit)
  compose_env.py - Gymnasium composition MDP (predict category x bucket)
  policy.py  - minimal numpy REINFORCE policy over tool selection (learnable)
  train.py   - REINFORCE training loop producing an owned checkpoint artifact
  train_sb3.py - SB3 PPO training loop (classic RL track)
  dataset.py - RLVR/SFT dataset export seam for LLM post-training (TRL/GRPO)
"""

from finrl.rl.dataset import export_rlvr_dataset
from finrl.rl.gym_env import (
    MASKED_TOOLS,
    MAX_ORDERS,
    TOOLS,
    Rule605GymEnv,
)
from finrl.rl.policy import SoftmaxToolPolicy
from finrl.rl.reward import confusion_matrix, dense_report_reward

__all__ = [
    "TOOLS",
    "MASKED_TOOLS",
    "MAX_ORDERS",
    "Rule605GymEnv",
    "SoftmaxToolPolicy",
    "confusion_matrix",
    "dense_report_reward",
    "export_rlvr_dataset",
]


def _register_envs() -> None:
    try:
        import gymnasium as gym
    except Exception:
        return
    for env_id, kwargs in (
        ("finrl/Rule605Tool-v0", {"action_set": "masked"}),
        ("finrl/Rule605ToolFull-v0", {"action_set": "full"}),
    ):
        try:
            if env_id in gym.registry:
                continue
            gym.register(id=env_id, entry_point="finrl.rl.gym_env:Rule605GymEnv", kwargs=kwargs)
        except Exception:
            continue
    try:
        from finrl.rl import compose_env  # noqa: F401
    except Exception:
        pass


_register_envs()
