"""RL package: turns the Rule 605 harness into a trainable MDP.

Components:
  reward.py  - dense, per-column shaped reward (verifiable reward for RLVR)
  gym_env.py - Gymnasium-compatible wrapper with intermediate shaping
  policy.py  - minimal numpy REINFORCE policy over tool selection (learnable)
  train.py   - training loop producing an owned checkpoint artifact
  dataset.py - RLVR/SFT dataset export seam for LLM post-training (TRL/GRPO)
"""

from finrl.rl.dataset import export_rlvr_dataset
from finrl.rl.gym_env import TOOLS, Rule605GymEnv
from finrl.rl.policy import SoftmaxToolPolicy
from finrl.rl.reward import confusion_matrix, dense_report_reward

__all__ = [
    "TOOLS",
    "Rule605GymEnv",
    "SoftmaxToolPolicy",
    "confusion_matrix",
    "dense_report_reward",
    "export_rlvr_dataset",
]
