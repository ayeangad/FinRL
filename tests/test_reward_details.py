"""Reward shaping + dense reward contract tests (regression guards)."""

from pathlib import Path

import finrl.rl  # noqa: F401
from finrl.rl.gym_env import (
    FIRST_EVIDENCE_BONUS,
    INVALID_PENALTY,
    REDUNDANT_PENALTY,
    STEP_COST,
    TRUNCATION_PENALTY,
    Rule605GymEnv,
)
from finrl.rl.reward import confusion_matrix, dense_report_reward

GOLDEN = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_shaping_constants_signs():
    assert FIRST_EVIDENCE_BONUS > 0
    assert REDUNDANT_PENALTY < 0
    assert INVALID_PENALTY < FIRST_EVIDENCE_BONUS
    assert STEP_COST < 0
    assert TRUNCATION_PENALTY < 0


def test_first_evidence_net_positive():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    _, r, _, _, _ = env.step(0, order_id="O1")
    assert abs(r - (STEP_COST + FIRST_EVIDENCE_BONUS)) < 1e-9


def test_redundant_net_negative_but_above_invalid():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    env.step(0, order_id="O1")
    _, r2, _, _, _ = env.step(0, order_id="O1")
    assert r2 == STEP_COST + REDUNDANT_PENALTY
    _, r3, _, _, _ = env.step(0, order_id="NOPE")
    assert r3 == STEP_COST + INVALID_PENALTY
    assert r3 < r2


def test_dense_weights_documented():
    # Regulatory (0.45) dominates numeric (0.35) dominates coverage (0.15).
    import inspect

    from finrl.rl import reward as _rw

    src = inspect.getsource(_rw.dense_report_reward)
    assert "0.45" in src and "0.35" in src and "0.15" in src


def test_dense_bounded():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    for bad in (None, "", "garbage", "a|b\nc|d"):
        out = dense_report_reward(bad, env.env.ground_truth_pipe)
        assert 0.0 <= out["reward"] <= 1.0


def test_confusion_counts_rows():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    cm = confusion_matrix(env.env.ground_truth_pipe, env.env.ground_truth_pipe)
    assert cm["missing"] == 0
    assert cm["spurious"] == 0
    assert cm["true_positives"] == cm["total_gt_rows"] >= 1
