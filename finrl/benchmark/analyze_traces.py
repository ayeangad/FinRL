import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from finrl.benchmark.evaluator import evaluate_submission
from finrl.benchmark.trace import AgentTrace


class ScenarioFailureAnalysis(BaseModel):
    scenario_id: str
    success: bool
    primary_failure_mode: str
    contributing_failure_modes: list[str] = Field(default_factory=list)
    score: float
    critical_errors: int
    major_errors: int
    minor_errors: int


class BaselineReport(BaseModel):
    title: str = "FinRL Phase 9 Zero-Shot Baseline Report"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str
    model_revision: str
    prompt_version: str
    scenario_set_version: str = "v0.1"
    total_scenarios: int
    success_rate: float
    average_score: float
    critical_error_rate: float
    major_error_rate: float
    minor_error_rate: float
    avg_numeric_accuracy: float
    avg_regulatory_accuracy: float
    avg_workflow_integrity: float
    failure_taxonomy: dict[str, int] = Field(default_factory=dict)
    per_scenario_results: list[ScenarioFailureAnalysis] = Field(default_factory=list)


def classify_trace_failures(trace: AgentTrace) -> ScenarioFailureAnalysis:
    gt_pipe = trace.evaluation.get("ground_truth_pipe", "")
    sub_pipe = trace.submission

    eval_detail = evaluate_submission(sub_pipe, gt_pipe)

    contributing: list[str] = []

    # Check for specific failure signals in step history
    has_syntax_error = any(s.action is None for s in trace.steps)
    has_tool_arg_error = any(
        s.tool_result and "Error" in str(s.tool_result.get("output", ""))
        for s in trace.steps
    )
    
    # Check premature submission: submitted report without running inspect tools
    called_inspect_tools = any(
        s.action and s.action.get("tool_name") in ("classify_order", "calculate_metrics", "get_order")
        for s in trace.steps
    )
    is_premature = sub_pipe is not None and not called_inspect_tools

    # Check step budget
    step_budget_exceeded = len(trace.steps) >= 50 and sub_pipe is None

    if has_syntax_error:
        contributing.append("ReAct Syntax Error")
    if has_tool_arg_error:
        contributing.append("Tool Argument Error")
    if is_premature:
        contributing.append("Premature Submission")
    if eval_detail.critical_errors > 0:
        contributing.append("Regulatory Misclassification")
    if eval_detail.major_errors > 0 or eval_detail.minor_errors > 0:
        contributing.append("Metric Discrepancy")
    if step_budget_exceeded:
        contributing.append("Step Budget Exceeded")

    # Determine Primary Failure Mode
    if eval_detail.success:
        primary = "SUCCESS"
        contributing = []
    elif step_budget_exceeded:
        primary = "Step Budget Exceeded"
    elif has_syntax_error and not sub_pipe:
        primary = "ReAct Syntax Error"
    elif is_premature:
        primary = "Premature Submission"
    elif eval_detail.critical_errors > 0:
        primary = "Regulatory Misclassification"
    elif has_tool_arg_error:
        primary = "Tool Argument Error"
    elif eval_detail.major_errors > 0 or eval_detail.minor_errors > 0:
        primary = "Metric Discrepancy"
    else:
        primary = "Metric Discrepancy"

    return ScenarioFailureAnalysis(
        scenario_id=trace.scenario_id,
        success=eval_detail.success,
        primary_failure_mode=primary,
        contributing_failure_modes=contributing,
        score=eval_detail.score,
        critical_errors=eval_detail.critical_errors,
        major_errors=eval_detail.major_errors,
        minor_errors=eval_detail.minor_errors,
    )


def generate_baseline_report(
    traces_dir: Path | str,
    output_dir: Path | str = "experiments/phase_9",
) -> BaselineReport:
    t_dir = Path(traces_dir)
    trace_files = sorted(list(t_dir.glob("*.json")))

    if not trace_files:
        raise ValueError(f"No trace JSON files found in {t_dir}")

    traces = [AgentTrace.model_validate_json(f.read_text()) for f in trace_files]
    total = len(traces)

    scenario_analyses = [classify_trace_failures(t) for t in traces]

    taxonomy = {
        "SUCCESS": 0,
        "ReAct Syntax Error": 0,
        "Tool Argument Error": 0,
        "Premature Submission": 0,
        "Regulatory Misclassification": 0,
        "Metric Discrepancy": 0,
        "Step Budget Exceeded": 0,
    }

    for sa in scenario_analyses:
        taxonomy[sa.primary_failure_mode] = taxonomy.get(sa.primary_failure_mode, 0) + 1

    success_count = sum(1 for sa in scenario_analyses if sa.success)
    avg_score = round(sum(sa.score for sa in scenario_analyses) / total, 4)

    first_trace = traces[0]
    report = BaselineReport(
        model_name=first_trace.model_name,
        model_revision=first_trace.model_revision,
        prompt_version=first_trace.prompt_version,
        total_scenarios=total,
        success_rate=round(success_count / total, 4),
        average_score=avg_score,
        critical_error_rate=round(sum(sa.critical_errors for sa in scenario_analyses) / total, 4),
        major_error_rate=round(sum(sa.major_errors for sa in scenario_analyses) / total, 4),
        minor_error_rate=round(sum(sa.minor_errors for sa in scenario_analyses) / total, 4),
        avg_numeric_accuracy=0.0,
        avg_regulatory_accuracy=0.0,
        avg_workflow_integrity=0.0,
        failure_taxonomy=taxonomy,
        per_scenario_results=scenario_analyses,
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = out_p / "baseline_report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    # Save Markdown report
    md_content = f"""# FinRL Phase 9 Zero-Shot Baseline Report

- **Model Name**: `{report.model_name}`
- **Model Revision**: `{report.model_revision}`
- **Prompt Version**: `{report.prompt_version}`
- **Total Scenarios**: `{report.total_scenarios}`
- **Generated At**: `{report.generated_at}`

---

## Executive Summary

| Metric | Baseline Value |
| :--- | :--- |
| **Success Rate** | **{report.success_rate * 100:.1f}%** |
| **Average Score** | **{report.average_score:.4f}** |
| **Critical Errors / Task** | `{report.critical_error_rate:.2f}` |
| **Major Errors / Task** | `{report.major_error_rate:.2f}` |
| **Minor Errors / Task** | `{report.minor_error_rate:.2f}` |

---

## Failure Mode Taxonomy

| Failure Category | Count | Percentage |
| :--- | ---: | ---: |
| **SUCCESS** | {taxonomy.get('SUCCESS', 0)} | {taxonomy.get('SUCCESS', 0) / total * 100:.1f}% |
| **ReAct Syntax Error** | {taxonomy.get('ReAct Syntax Error', 0)} | {taxonomy.get('ReAct Syntax Error', 0) / total * 100:.1f}% |
| **Tool Argument Error** | {taxonomy.get('Tool Argument Error', 0)} | {taxonomy.get('Tool Argument Error', 0) / total * 100:.1f}% |
| **Premature Submission** | {taxonomy.get('Premature Submission', 0)} | {taxonomy.get('Premature Submission', 0) / total * 100:.1f}% |
| **Regulatory Misclassification** | {taxonomy.get('Regulatory Misclassification', 0)} | {taxonomy.get('Regulatory Misclassification', 0) / total * 100:.1f}% |
| **Metric Discrepancy** | {taxonomy.get('Metric Discrepancy', 0)} | {taxonomy.get('Metric Discrepancy', 0) / total * 100:.1f}% |
| **Step Budget Exceeded** | {taxonomy.get('Step Budget Exceeded', 0)} | {taxonomy.get('Step Budget Exceeded', 0) / total * 100:.1f}% |
"""
    md_path = out_p / "baseline_report.md"
    md_path.write_text(md_content)

    return report
