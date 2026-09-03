from pathlib import Path

from finrl.env import Rule605Env, ToolAction

GOLDEN_SCENARIO = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_env_reset_and_observation():
    env = Rule605Env()
    obs = env.reset(GOLDEN_SCENARIO)

    assert obs.scenario_id == "market_01"
    assert obs.security == "FINRL"
    assert len(obs.orders) == 1
    assert obs.quotes_count == 1
    assert obs.executions_count == 1
    assert obs.current_step == 0
    assert obs.action_history == []


def test_env_tool_interactions():
    env = Rule605Env()
    env.reset(GOLDEN_SCENARIO)

    # 1. get_order
    res1 = env.step(ToolAction(tool_name="get_order", arguments={"order_id": "O1"}))
    assert res1.info["tool_output"]["order_id"] == "O1"
    assert not res1.done

    # 2. get_executions
    res2 = env.step(ToolAction(tool_name="get_executions", arguments={"order_id": "O1"}))
    assert len(res2.info["tool_output"]) == 1
    assert res2.info["tool_output"][0]["execution_id"] == "E1"

    # 3. classify_order
    res3 = env.step(ToolAction(tool_name="classify_order", arguments={"order_id": "O1"}))
    assert res3.info["tool_output"]["order_type_category"] == "market"
    assert res3.info["tool_output"]["order_size_bucket"] == "100_to_499"
    assert res3.info["tool_output"]["reportable"] is True

    # 4. calculate_metrics
    res4 = env.step(ToolAction(tool_name="calculate_metrics", arguments={"order_id": "O1"}))
    assert res4.info["tool_output"]["order_id"] == "O1"
    assert res4.info["tool_output"]["effective_spread"] == "0.20"

    assert env.current_step == 4
    assert len(env.action_history) == 4


def test_env_submit_report_exact_match():
    env = Rule605Env()
    env.reset(GOLDEN_SCENARIO)

    # Get ground truth pipe string from environment
    gt_pipe = env.ground_truth_pipe

    res = env.step(ToolAction(tool_name="submit_report", arguments={"report": gt_pipe}))
    assert res.done
    assert res.reward == 1.0
    assert res.info["exact_match"] is True


def test_env_submit_report_incorrect():
    env = Rule605Env()
    env.reset(GOLDEN_SCENARIO)

    res = env.step(ToolAction(tool_name="submit_report", arguments={"report": "invalid_pipe_data"}))
    assert res.done
    assert res.reward < 1.0
    assert res.info["exact_match"] is False


def test_env_max_steps_truncation():
    env = Rule605Env(max_steps=2)
    env.reset(GOLDEN_SCENARIO)

    res1 = env.step(ToolAction(tool_name="get_order", arguments={"order_id": "O1"}))
    assert not res1.done

    res2 = env.step(ToolAction(tool_name="get_order", arguments={"order_id": "O1"}))
    assert res2.done
    assert res2.info.get("truncated") is True
