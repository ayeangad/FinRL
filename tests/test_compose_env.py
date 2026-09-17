"""Compose env tests: prediction learning, grounding gate, submit assembly."""

from pathlib import Path

import numpy as np

import finrl.rl  # noqa: F401
from finrl.rl.compose_env import N_COMBOS, Rule605ComposeEnv

GOLDEN = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_compose_spaces():
    env = Rule605ComposeEnv(max_steps=10)
    obs, _ = env.reset(GOLDEN, seed=0)
    assert set(obs.keys()) == {
        "n_orders", "step", "coverage", "order_features",
        "grounded_mask", "pred_mask", "pred_combo",
    }
    assert list(env.action_space.nvec) == [2, 32, 30]
    assert env.observation_space.contains(obs)


def test_compose_check_env():
    from gymnasium.utils.env_checker import check_env

    check_env(Rule605ComposeEnv(max_steps=10).unwrapped, skip_render_check=True)


def test_correct_predict_bonus():
    env = Rule605ComposeEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    gt_combo = list(env._gt_combo.values())[0]
    _, r, _, _, info = env.step([1, 0, gt_combo])
    assert r > 0
    assert info["correct"] is True


def test_wrong_predict_penalty():
    env = Rule605ComposeEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    gt_combo = list(env._gt_combo.values())[0]
    wrong = (gt_combo + 1) % N_COMBOS
    _, r, _, _, info = env.step([1, 0, wrong])
    assert r < 0
    assert info["correct"] is False


def test_grounding_gate():
    # Predict correctly but never ground -> submit must NOT score 1.0.
    env = Rule605ComposeEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    gt_combo = list(env._gt_combo.values())[0]
    env.step([1, 0, gt_combo])
    _, r, _, _, info = env.step([0, 0, 4])  # submit
    assert r < 1.0


def test_ground_predict_submit_scores_one():
    env = Rule605ComposeEnv(max_steps=12)
    env.reset(GOLDEN, seed=0)
    gt_combo = list(env._gt_combo.values())[0]
    env.step([0, 0, 0])  # get_order grounds slot 0
    env.step([1, 0, gt_combo])
    _, r, term, _, info = env.step([0, 0, 4])
    assert r == 1.0
    assert term is True


def test_invalid_slot_penalty():
    env = Rule605ComposeEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    _, r, _, _, _ = env.step([1, 31, 0])  # slot 31 invalid for 1-order scenario
    assert r < 0


def test_combo_roundtrip():
    from finrl.rl.compose_env import cat_bucket_to_combo, combo_to_cat_bucket

    for c in range(N_COMBOS):
        cat, bucket = combo_to_cat_bucket(c)
        assert cat_bucket_to_combo(cat, bucket) == c
