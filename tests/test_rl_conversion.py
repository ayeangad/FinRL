from pathlib import Path

from finrl.env.rule_605_env import Rule605Env
from finrl.rl.gym_env import Rule605GymEnv
from finrl.rl.policy import SoftmaxToolPolicy
from finrl.rl.policy_agent import RLToolAgent
from finrl.rl.reward import confusion_matrix, dense_report_reward

GOLDEN = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_dense_reward_exact_match_is_one():
    env = Rule605Env()
    env.reset(GOLDEN)
    out = dense_report_reward(env.ground_truth_pipe, env.ground_truth_pipe)
    assert out["reward"] == 1.0
    assert out["row_coverage"] == 1.0
    assert out["regulatory_accuracy"] == 1.0
    assert all(v == 1.0 for v in out["per_column_accuracy"].values())


def test_dense_reward_empty_is_zero():
    env = Rule605Env()
    env.reset(GOLDEN)
    out = dense_report_reward(None, env.ground_truth_pipe)
    assert out["reward"] == 0.0


def test_dense_reward_partial_credit_in_between():
    env = Rule605Env()
    env.reset(GOLDEN)
    gt = env.ground_truth_pipe
    header = gt.split("\n")[0]
    out = dense_report_reward(header, gt)
    assert 0.0 < out["reward"] < 1.0
    assert out["matched_rows"] == 0


def test_dense_reward_numeric_tolerance():
    env = Rule605Env()
    env.reset(GOLDEN)
    gt = env.ground_truth_pipe
    lines = gt.split("\n")
    # Perturb a float metric by 1e-9 -> within default tolerance -> still 1.0
    parts = lines[1].split("|")
    try:
        val = float(parts[10])
        parts[10] = str(val + 1e-9)
        mod = "\n".join([lines[0], "|".join(parts), *lines[2:]])
        out = dense_report_reward(mod, gt)
        assert out["reward"] == 1.0
    except (ValueError, IndexError):
        pass


def test_confusion_matrix_exact():
    env = Rule605Env()
    env.reset(GOLDEN)
    cm = confusion_matrix(env.ground_truth_pipe, env.ground_truth_pipe)
    assert cm["missing"] == 0
    assert cm["true_positives"] == cm["total_gt_rows"]


def test_gym_env_shaping_and_terminal():
    gym = Rule605GymEnv(max_steps=10)  # masked: 0=classify, 1=metrics, 2=submit
    gym.reset(GOLDEN)
    # First classify: bonus + step cost = net positive
    _, r1, *_ = gym.step(0, order_id="O1")
    assert r1 > 0
    # Redundant classify: net negative
    _, r2, *_ = gym.step(0, order_id="O1")
    assert r2 < r1
    # Metrics then submit -> dense 1.0
    gym.step(1, order_id="O1")
    out = gym.step(2)
    reward = out[1]
    assert reward == 1.0


def test_gym_env_full_action_set():
    gym = Rule605GymEnv(max_steps=10, action_set="full")
    assert gym.action_space_n == 7
    gym.reset(GOLDEN)
    _, r1, *_ = gym.step(4, order_id="O1")
    assert r1 > 0


def test_reinforce_update_changes_weights(tmp_path):
    pol = SoftmaxToolPolicy(seed=0)
    before = pol.W.copy()
    trajs = [
        {"features": [[1, 0.03, 0.0, 0.0, 0.0]] * 3, "actions": [0, 1, 2], "rewards": [0.045, 0.045, 1.0]},
        {"features": [[1, 0.03, 0.0, 0.0, 0.0]] * 2, "actions": [0, 0], "rewards": [-0.005, -0.1]},
    ]
    stats = pol.reinforce_update(trajs, lr=0.1)
    assert (pol.W != before).any()
    assert stats["episodes"] == 2
    ckpt = pol.save(tmp_path / "pol.npz")
    loaded = SoftmaxToolPolicy.load(ckpt)
    assert (loaded.W == pol.W).all()


def test_rl_tool_agent_scores_one():
    from finrl.benchmark.evaluator import evaluate_submission

    env = Rule605Env()
    env.reset(GOLDEN)
    agent = RLToolAgent()
    traj = agent.run(env)
    assert traj.submitted_pipe is not None
    detail = evaluate_submission(traj.submitted_pipe, env.ground_truth_pipe)
    assert detail.score == 1.0
    assert detail.success is True


def test_rl_tool_agent_trained_checkpoint_scores_one():
    from finrl.benchmark.evaluator import evaluate_submission

    ckpt = Path(__file__).parent.parent / "checkpoints" / "tool_policy.npz"
    assert ckpt.exists(), "train first: python -m finrl.rl.train --out checkpoints/tool_policy.npz"
    env = Rule605Env()
    env.reset(GOLDEN)
    agent = RLToolAgent(checkpoint=ckpt)
    traj = agent.run(env)
    assert traj.submitted_pipe is not None
    detail = evaluate_submission(traj.submitted_pipe, env.ground_truth_pipe)
    assert detail.score == 1.0
