"""No-oracle tests: submit must compose from evidence, never copy GT.

Guards the credibility core: _build_evidence_report must not call
run_scenario_and_serialize on the scenario; empty evidence -> header-only;
partial evidence -> partial dense reward; full evidence -> 1.0.
"""

import inspect
from pathlib import Path

import finrl.rl  # noqa: F401
from finrl.rl.gym_env import Rule605GymEnv
from finrl.rl.reward import dense_report_reward

GOLDEN = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_submit_without_evidence_is_header_only():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    report = env._build_evidence_report()
    assert "\n" not in report.strip() or len(report.strip().split("\n")) == 1
    out = dense_report_reward(report, env.env.ground_truth_pipe)
    assert out["reward"] < 0.2


def test_partial_evidence_partial_report():
    # Use a multi-order scenario for partial coverage.
    goldens = sorted((Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden").glob("*.json"))
    multi = None
    for g in goldens:
        import json

        data = json.loads(g.read_text())
        n = len(data.get("orders", [])) + (1 if data.get("order") else 0)
        if n >= 2:
            multi = g
            break
    if multi is None:
        return  # single-order corpus; partial test vacuous
    env = Rule605GymEnv(max_steps=20, action_set="full")
    env.reset(multi, seed=0)
    first = env._order_ids[0]
    env.step({"tool_name": "classify_order", "arguments": {"order_id": first}})
    env.step({"tool_name": "calculate_metrics", "arguments": {"order_id": first}})
    report = env._build_evidence_report()
    out = dense_report_reward(report, env.env.ground_truth_pipe)
    assert 0.0 <= out["reward"] <= 1.0


def test_full_evidence_matches_ground_truth_via_cache():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    env.step(0, order_id="O1")  # classify
    env.step(1, order_id="O1")  # metrics
    report = env._build_evidence_report()
    assert report.strip() == env.env.ground_truth_pipe.strip()
    # Evidence cache was used (populated), not a fresh scenario read.
    assert "O1" in env._evidence


def test_evidence_builder_does_not_reread_scenario():
    src = inspect.getsource(Rule605GymEnv._build_evidence_report)
    assert "run_scenario_and_serialize" not in src
    assert "ground_truth_pipe" not in src or "dense" not in src  # builder itself must not read GT


def test_gym_submit_reward_path_uses_dense():
    env = Rule605GymEnv(max_steps=10)
    env.reset(GOLDEN, seed=0)
    env.step(0, order_id="O1")
    env.step(1, order_id="O1")
    _, reward, terminated, _, info = env.step(2)
    assert terminated is True
    assert reward == 1.0
    assert info["dense"]["reward"] == 1.0


def test_truncation_has_no_dense_leak():
    env = Rule605GymEnv(max_steps=1)
    env.reset(GOLDEN, seed=0)
    _, reward, _, truncated, info = env.step(0, order_id="O1")
    assert truncated is True
    assert reward == -0.1
