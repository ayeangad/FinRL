"""Gymnasium API compliance tests for Rule605GymEnv.

Covers: gym.make registration, Dict/MultiDiscrete spaces, 5-tuple step,
seeding determinism, default reset(), SB3 VecEnv compatibility surface.
"""

from pathlib import Path

import numpy as np

import finrl.rl  # noqa: F401  (registers finrl/* envs)
from finrl.rl.gym_env import MAX_ORDERS, ORDER_FEATURE_DIM, Rule605GymEnv

GOLDEN = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_gym_make_registration():
    gym = __import__("gymnasium")
    for env_id in ("finrl/Rule605Tool-v0", "finrl/Rule605ToolFull-v0"):
        env = gym.make(env_id, max_steps=10)
        assert env.action_space is not None
        assert env.observation_space is not None
        env.close()


def test_observation_space_keys_and_shapes():
    env = Rule605GymEnv(max_steps=10)
    obs, info = env.reset(GOLDEN, seed=0)
    assert set(obs.keys()) == {"n_orders", "step", "coverage", "order_features", "evidence_mask"}
    assert obs["order_features"].shape == (MAX_ORDERS, ORDER_FEATURE_DIM)
    assert obs["evidence_mask"].shape == (MAX_ORDERS, 2)
    assert float(np.asarray(obs["n_orders"]).ravel()[0]) == 1.0
    # Spaces contain the observation.
    assert env.observation_space.contains(obs), "reset obs must be in observation_space"


def test_action_space_is_multidiscrete():
    gym = __import__("gymnasium")
    env = Rule605GymEnv(max_steps=10, action_set="masked")
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert list(env.action_space.nvec) == [3, MAX_ORDERS]
    env_full = Rule605GymEnv(max_steps=10, action_set="full")
    assert list(env_full.action_space.nvec) == [7, MAX_ORDERS]


def test_step_always_five_tuple():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=1)
    out = env.step([0, 0])  # classify slot 0
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert set(obs.keys()) == {"n_orders", "step", "coverage", "order_features", "evidence_mask"}


def test_legacy_int_action_still_works():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    _, r1, *_ = env.step(0, order_id="O1")
    assert r1 > 0
    _, r2, *_ = env.step(0, order_id="O1")
    assert r2 < r1


def test_seeding_determinism():
    env = Rule605GymEnv(max_steps=10)
    o1, _ = env.reset(GOLDEN, seed=123)
    env2 = Rule605GymEnv(max_steps=10)
    o2, _ = env2.reset(GOLDEN, seed=123)
    assert np.allclose(o1["order_features"], o2["order_features"])
    assert float(np.asarray(o1["coverage"]).ravel()[0]) == float(np.asarray(o2["coverage"]).ravel()[0])


def test_default_reset_no_args():
    env = Rule605GymEnv(max_steps=5)
    obs, info = env.reset(seed=0)
    assert "n_orders" in obs
    assert "scenario_id" in info


def test_gym_check_env():
    gym = __import__("gymnasium")
    from gymnasium.utils.env_checker import check_env

    env = Rule605GymEnv(max_steps=10).unwrapped
    check_env(env, skip_render_check=True)


def test_multidiscrete_submit_terminates():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    env.step([0, 0])  # classify
    env.step([1, 0])  # metrics
    obs, reward, terminated, truncated, info = env.step([2, 0])  # submit
    assert terminated is True
    assert reward == 1.0
    assert "dense" in info


def test_truncation_penalty():
    env = Rule605GymEnv(max_steps=2)
    env.reset(GOLDEN, seed=0)
    env.step([0, 0])
    _, reward, terminated, truncated, info = env.step([0, 0])
    assert truncated is True
    assert terminated is False
    assert reward == -0.1
