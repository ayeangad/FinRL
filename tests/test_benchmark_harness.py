from pathlib import Path

from finrl.benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    BrokenAgent,
    ReferenceAgent,
    evaluate_submission,
)

GOLDEN_DIR = Path(__file__).parent.parent / "scenarios" / "v0.1" / "golden"


def test_reference_agent_benchmark_100_percent_success():
    config = BenchmarkConfig(scenarios_dir=GOLDEN_DIR)
    runner = BenchmarkRunner(config)
    agent = ReferenceAgent()

    result = runner.run_benchmark(agent)

    assert result.total_scenarios == 100
    assert result.success_rate == 1.0
    assert result.average_score == 1.0
    assert result.critical_error_rate == 0.0
    assert result.major_error_rate == 0.0
    assert result.minor_error_rate == 0.0
    assert result.avg_regulatory_accuracy == 1.0
    assert result.avg_numeric_accuracy == 1.0
    assert result.avg_workflow_integrity == 1.0


def test_broken_agent_benchmark_detects_errors():
    config = BenchmarkConfig(scenarios_dir=GOLDEN_DIR)
    runner = BenchmarkRunner(config)
    agent = BrokenAgent()

    result = runner.run_benchmark(agent)

    assert result.total_scenarios == 100
    assert result.success_rate == 0.0
    assert result.average_score < 0.5
    assert result.critical_error_rate > 0.0


def test_structured_evaluator_severities():
    gt_pipe = (
        "order_type_category|order_size_bucket|num_covered_orders|num_executed_orders|cumulative_shares|cumulative_executed_shares|shares_price_improved|shares_at_quote|shares_outside_quote|price_improvement|effective_spread|quoted_spread|realized_spread_50ms|realized_spread_1s|realized_spread_15s|realized_spread_1m|realized_spread_5m|percentage_effective_spread|percentage_quoted_spread|percentage_realized_spread_50ms|percentage_realized_spread_1s|percentage_realized_spread_15s|percentage_realized_spread_1m|percentage_realized_spread_5m\n"
        "market|100_to_499|1|1|100|100|100|0|0|0.10|0.00|0.20||||||||||||"
    )

    # 1. Exact match
    res1 = evaluate_submission(gt_pipe, gt_pipe)
    assert res1.success
    assert res1.score == 1.0
    assert res1.critical_errors == 0

    # 2. Critical error: wrong regulatory count (num_covered_orders = 999)
    sub_critical = gt_pipe.replace("market|100_to_499|1|1|", "market|100_to_499|999|1|")
    res2 = evaluate_submission(sub_critical, gt_pipe)
    assert not res2.success
    assert res2.critical_errors >= 1

    # 3. Major error: wrong share volume (shares_price_improved = 50 instead of 100)
    sub_major = gt_pipe.replace("100|100|100|0|0|", "100|100|50|0|0|")
    res3 = evaluate_submission(sub_major, gt_pipe)
    assert res3.major_errors >= 1

    # 4. Minor error: wrong metric spread (price_improvement = 0.99 instead of 0.10)
    sub_minor = gt_pipe.replace("|0.10|0.00|0.20|", "|0.99|0.00|0.20|")
    res4 = evaluate_submission(sub_minor, gt_pipe)
    assert res4.minor_errors >= 1
