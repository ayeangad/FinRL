from pathlib import Path
import json

from finrl.benchmark.analyze_traces import (
    AgentTrace,
    classify_trace_failures,
    generate_baseline_report,
)
from finrl.benchmark.trace import StepTrace, save_trace


def test_classify_trace_failures_success():
    gt_pipe = (
        "order_type_category|order_size_bucket|num_covered_orders|num_executed_orders|cumulative_shares|cumulative_executed_shares|shares_price_improved|shares_at_quote|shares_outside_quote|price_improvement|effective_spread|quoted_spread|realized_spread_50ms|realized_spread_1s|realized_spread_15s|realized_spread_1m|realized_spread_5m|percentage_effective_spread|percentage_quoted_spread|percentage_realized_spread_50ms|percentage_realized_spread_1s|percentage_realized_spread_15s|percentage_realized_spread_1m|percentage_realized_spread_5m\n"
        "market|100_to_499|1|1|100|100|100|0|0|0.10|0.00|0.20||||||||||||"
    )
    trace = AgentTrace(
        trace_id="t1",
        scenario_id="market_01",
        model_name="qwen3_4b_mock",
        steps=[
            StepTrace(
                step=1,
                raw_model_output='Thought: Inspecting order\nAction: {"tool_name": "classify_order", "arguments": {"order_id": "O1"}}',
                action={"tool_name": "classify_order", "arguments": {"order_id": "O1"}},
            ),
            StepTrace(
                step=2,
                raw_model_output=f'Thought: Submit report\nAction: {{"tool_name": "submit_report", "arguments": {{"report": "{gt_pipe}"}}}}',
                action={"tool_name": "submit_report", "arguments": {"report": gt_pipe}},
            ),
        ],
        submission=gt_pipe,
        evaluation={"ground_truth_pipe": gt_pipe},
    )

    analysis = classify_trace_failures(trace)
    assert analysis.success is True
    assert analysis.primary_failure_mode == "SUCCESS"


def test_generate_baseline_report(tmp_path: Path):
    gt_pipe = "header\ndata"
    trace = AgentTrace(
        trace_id="t1",
        scenario_id="market_01",
        model_name="qwen3_4b_mock",
        steps=[],
        submission="wrong_data",
        evaluation={"ground_truth_pipe": gt_pipe},
    )

    traces_dir = tmp_path / "traces"
    save_trace(trace, base_dir=traces_dir)

    # Note: save_trace saves to traces_dir / prompt_version / model_name / scenario_id.json
    model_traces_dir = traces_dir / trace.prompt_version / trace.model_name
    output_dir = tmp_path / "experiments" / "phase_9"

    report = generate_baseline_report(traces_dir=model_traces_dir, output_dir=output_dir)

    assert report.total_scenarios == 1
    assert (output_dir / "baseline_report.json").exists()
    assert (output_dir / "baseline_report.md").exists()
