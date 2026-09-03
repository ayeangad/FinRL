import math
import time
from collections.abc import Sequence
from pathlib import Path

from finrl.benchmark.agent import BaseAgent
from finrl.benchmark.config import BenchmarkConfig
from finrl.benchmark.evaluator import evaluate_submission
from finrl.benchmark.result import BenchmarkResult, ScenarioResult
from finrl.env.rule_605_env import Rule605Env


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return round(d0 + d1, 2)


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()

    def run_benchmark(
        self,
        agent: BaseAgent,
        scenarios_dir: Path | str | None = None,
        limit: int | None = None,
    ) -> BenchmarkResult:
        s_dir = Path(scenarios_dir) if scenarios_dir else self.config.scenarios_dir
        scenario_files = sorted(s_dir.glob("*.json"))

        max_limit = limit if limit is not None else self.config.limit
        if max_limit is not None:
            if max_limit <= 0:
                raise ValueError("Limit must be a positive integer greater than 0")
            scenario_files = scenario_files[:max_limit]

        if not scenario_files:
            raise ValueError(f"No scenario .json files found in {s_dir}")

        scenario_results = []
        latencies = []

        for s_path in scenario_files:
            env = Rule605Env(max_steps=self.config.max_steps_per_task)
            env.reset(s_path)

            t0 = time.perf_counter()
            trajectory = agent.run(env)
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000.0, 2)
            latencies.append(latency_ms)

            eval_detail = evaluate_submission(
                submitted_pipe=trajectory.submitted_pipe,
                ground_truth_pipe=env.ground_truth_pipe,
            )

            res = ScenarioResult(
                scenario_id=s_path.stem,
                model_name=agent.name,
                success=eval_detail.success,
                score=eval_detail.score,
                critical_errors=eval_detail.critical_errors,
                major_errors=eval_detail.major_errors,
                minor_errors=eval_detail.minor_errors,
                numeric_accuracy=eval_detail.numeric_accuracy,
                regulatory_accuracy=eval_detail.regulatory_accuracy,
                evidence_accuracy=eval_detail.evidence_accuracy,
                workflow_integrity=eval_detail.workflow_integrity,
                tool_calls=trajectory.tool_calls_count,
                steps=trajectory.steps_count,
                invalid_actions=trajectory.invalid_actions_count,
                latency_ms=latency_ms,
                cost_usd=0.0,
            )
            scenario_results.append(res)

        total = len(scenario_results)
        success_count = sum(1 for r in scenario_results if r.success)

        return BenchmarkResult(
            model_name=agent.name,
            total_scenarios=total,
            success_rate=round(success_count / total, 4),
            average_score=round(sum(r.score for r in scenario_results) / total, 4),
            critical_error_rate=round(sum(r.critical_errors for r in scenario_results) / total, 4),
            major_error_rate=round(sum(r.major_errors for r in scenario_results) / total, 4),
            minor_error_rate=round(sum(r.minor_errors for r in scenario_results) / total, 4),
            avg_numeric_accuracy=round(sum(r.numeric_accuracy for r in scenario_results) / total, 4),
            avg_regulatory_accuracy=round(sum(r.regulatory_accuracy for r in scenario_results) / total, 4),
            avg_evidence_accuracy=round(sum(r.evidence_accuracy for r in scenario_results) / total, 4),
            avg_workflow_integrity=round(sum(r.workflow_integrity for r in scenario_results) / total, 4),
            avg_tool_calls=round(sum(r.tool_calls for r in scenario_results) / total, 2),
            avg_steps=round(sum(r.steps for r in scenario_results) / total, 2),
            total_invalid_actions=sum(r.invalid_actions for r in scenario_results),
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            p99_latency_ms=_percentile(latencies, 99),
            total_cost_usd=round(sum(r.cost_usd for r in scenario_results), 4),
            scenario_results=scenario_results,
        )
