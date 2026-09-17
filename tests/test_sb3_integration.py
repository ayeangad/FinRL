"""SB3 integration: PPO can train on both envs (smoke, CPU-fast)."""

import finrl.rl  # noqa: F401


def test_ppo_tool_smoke(tmp_path):
    from stable_baselines3 import PPO

    import gymnasium as gym

    env = gym.make("finrl/Rule605Tool-v0", max_steps=8)
    model = PPO("MultiInputPolicy", env, seed=0, verbose=0)
    model.learn(total_timesteps=512)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert len(action) == 2
    env.close()


def test_ppo_compose_smoke(tmp_path):
    from stable_baselines3 import PPO

    import gymnasium as gym

    env = gym.make("finrl/Rule605Compose-v0", max_steps=8)
    model = PPO("MultiInputPolicy", env, seed=0, verbose=0)
    model.learn(total_timesteps=512)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert len(action) == 3
    env.close()


def test_evaluate_random_smoke():
    from finrl.rl.evaluate import run_eval

    res = run_eval(env="tool", split="val", episodes=1, policies=["random"], max_steps=8, out="/tmp/eval_test_smoke.json")
    assert "random" in res["policies"]
    assert 0.0 <= res["policies"]["random"]["mean_dense"] <= 1.0


def test_train_sb3_fn_smoke(tmp_path):
    from finrl.rl.train_sb3 import train_ppo

    out = tmp_path / "ppo_smoke.zip"
    res = train_ppo(
        env_id="finrl/Rule605Tool-v0",
        timesteps=1024,
        seed=0,
        n_envs=2,
        max_steps=8,
        out=out,
        eval_freq=1024,
    )
    assert out.exists()
    assert res["timesteps"] == 1024
