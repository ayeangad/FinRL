from pathlib import Path
import json

from finrl.benchmark import Qwen4BAgent, AgentTrace, StepTrace, parse_react_output, save_trace
from finrl.env import Rule605Env
from finrl.models.qwen3_4b import Qwen3_4B_Runner

GOLDEN_SCENARIO = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden" / "market_01.json"


def test_react_parser_valid_json():
    raw = (
        "Thought: I need to classify order O1 against receipt NBBO.\n"
        'Action: {"tool_name": "classify_order", "arguments": {"order_id": "O1"}}'
    )
    res = parse_react_output(raw)
    assert res.error is None
    assert res.thought == "I need to classify order O1 against receipt NBBO."
    assert res.tool_action is not None
    assert res.tool_action.tool_name == "classify_order"
    assert res.tool_action.arguments == {"order_id": "O1"}


def test_react_parser_invalid_json():
    raw = (
        "Thought: Let me check quotes.\n"
        'Action: {"tool_name": "get_quote", "arguments": invalid_json}'
    )
    res = parse_react_output(raw)
    assert res.error is not None
    assert res.tool_action is None


def test_trace_schema_and_saving(tmp_path: Path):
    trace = AgentTrace(
        trace_id="test_123",
        scenario_id="market_01",
        model_name="qwen3_4b",
        model_revision="qwen3-4b-instruct",
        prompt_version="rule_605_v1",
        steps=[
            StepTrace(
                step=1,
                raw_model_output='Thought: Test\nAction: {"tool_name": "classify_order", "arguments": {"order_id": "O1"}}',
                thought="Test",
                action={"tool_name": "classify_order", "arguments": {"order_id": "O1"}},
                tool_result={"output": {"reportable": True}},
            )
        ],
        submission="pipe|data",
        evaluation={"score": 1.0},
    )

    saved_path = save_trace(trace, base_dir=tmp_path)
    assert saved_path.exists()

    data = json.loads(saved_path.read_text())
    assert data["trace_id"] == "test_123"
    assert data["steps"][0]["raw_model_output"].startswith("Thought: Test")


def test_qwen_agent_single_scenario_end_to_end(tmp_path: Path):
    runner = Qwen3_4B_Runner(mode="mock")
    agent = Qwen4BAgent(mode="mock", runner=runner, traces_dir=tmp_path)

    env = Rule605Env()
    env.reset(GOLDEN_SCENARIO)

    trajectory = agent.run(env)

    assert env.done
    assert trajectory.agent_name == "qwen3_4b_mock"
    assert trajectory.scenario_id == "market_01"
    assert trajectory.steps_count >= 1
    assert trajectory.submitted_pipe is not None

    # Check trace saved
    expected_trace_file = tmp_path / "rule_605_v1" / "qwen3_4b_mock" / "market_01.json"
    assert expected_trace_file.exists()
