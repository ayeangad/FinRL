"""Observation encoding tests: features, padding, masks, determinism."""

from pathlib import Path

import numpy as np

import finrl.rl  # noqa: F401
from finrl.rl.gym_env import MAX_ORDERS, ORDER_FEATURE_DIM, Rule605GymEnv

GOLDEN = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_order_features_market_buy():
    env = Rule605GymEnv(max_steps=10)
    obs, _ = env.reset(GOLDEN, seed=0)
    f = obs["order_features"][0]
    assert len(f) == ORDER_FEATURE_DIM
    assert f[0] == 1.0  # buy
    assert f[1] == 1.0  # market one-hot
    assert f[2] == f[3] == f[4] == 0.0
    assert 0.0 <= f[5] <= 1.0  # log qty
    assert f[6] == 0.0 and f[7] == 0.0  # no limit/stop


def test_padding_is_zero():
    env = Rule605GymEnv(max_steps=10)
    obs, _ = env.reset(GOLDEN, seed=0)
    n = int(np.asarray(obs["n_orders"]).ravel()[0])
    assert n == 1
    assert np.allclose(obs["order_features"][n:], 0.0)
    assert np.allclose(obs["evidence_mask"][n:], 0.0)


def test_evidence_mask_tracks_tools():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    obs, _, _, _, _ = env.step(0, order_id="O1")
    assert obs["evidence_mask"][0, 0] == 1.0
    assert obs["evidence_mask"][0, 1] == 0.0
    obs, _, _, _, _ = env.step(1, order_id="O1")
    assert obs["evidence_mask"][0, 1] == 1.0
    assert float(np.asarray(obs["coverage"]).ravel()[0]) == 1.0


def test_multidiscrete_slot_wraps_to_valid_order():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    # Slot 31 wraps to O1 for single-order scenarios (no invalid penalty).
    _, r, _, _, _ = env.step([0, 31])
    assert r > 0


def test_reset_seed_changes_nothing_structural():
    env = Rule605GymEnv(max_steps=10)
    o1, _ = env.reset(GOLDEN, seed=1)
    o2, _ = env.reset(GOLDEN, seed=999)
    assert np.allclose(o1["order_features"], o2["order_features"])
