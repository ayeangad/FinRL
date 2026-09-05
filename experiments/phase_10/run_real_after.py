"""Run the real Qwen3-0.6B agent on a diverse scenario subset.

Uses the phase-10 "after" system: scenario-aware selected rule sections +
windowed conversation history. Traces go under traces/phase_10/scenario_aware/.

The "before" baseline is documented by the phase-9 openai_real traces (context
grew every step; e.g. edge_case_10 reached ~14,198 input tokens by step 9)
and reproduced offline by experiments/phase_10/validate_context.py.

Run:  .venv/bin/python -m experiments.phase_10.run_real_after
"""

import json
import time
from pathlib import Path

from finrl.benchmark.agent import QwenAgent
from finrl.benchmark.evaluator import evaluate_submission
from finrl.env.rule_605_env import Rule605Env

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "scenarios" / "v0.1" / "golden"
TRACES_ROOT = REPO / "traces" / "phase_10"

STRATEGIES = ["scenario_aware"]

SCENARIOS = [
    "market_01.json",
    "market_05.json",
    "stop_01.json",
    "marketable_limit_01.json",
    "midpoint_limit_01.json",
    "edge_case_07.json",
]


def run_one(scenario: str, strategy: str) -> dict:
    env = Rule605Env(max_steps=50)
    env.reset(GOLDEN / scenario)

    traces_dir = TRACES_ROOT / strategy
    agent = QwenAgent(
        mode="real",
        traces_dir=traces_dir,
        use_scenario_context=(strategy == "scenario_aware"),
        max_history_tokens=3000,
        keep_last_n_exchanges=3,
    )

    t0 = time.perf_counter()
    traj = agent.run(env)
    elapsed = time.perf_counter() - t0

    eval_detail = evaluate_submission(
        submitted_pipe=traj.submitted_pipe,
        ground_truth_pipe=env.ground_truth_pipe,
    )

    trace_file = traces_dir / "rule_605_v1" / agent.name / f"{scenario[:-5]}.json"
    trace = json.loads(trace_file.read_text())
    ctx = trace.get("metrics", {}).get("context", {})

    return {
        "scenario": scenario,
        "strategy": strategy,
        "steps": traj.steps_count,
        "invalid_actions": traj.invalid_actions_count,
        "submitted": traj.submitted_pipe is not None,
        "score": eval_detail.score,
        "success": eval_detail.success,
        "elapsed_s": round(elapsed, 1),
        "input_tokens": trace.get("metrics", {}).get("input_tokens", 0),
        "output_tokens": trace.get("metrics", {}).get("output_tokens", 0),
        "full_history_est": ctx.get("history_full_est_tokens", 0),
        "bounded_history_est": ctx.get("history_bounded_est_tokens", 0),
        "dropped_exchanges": ctx.get("dropped_exchanges", 0),
        "sections": ctx.get("selected_sections", []),
    }


def main() -> None:
    out = REPO / "experiments" / "phase_10" / "real_ab.txt"
    lines = []

    print("=" * 96)
    print("REAL Qwen3-0.6B 'after': scenario-aware sections + windowed history (phase-10)")
    print("=" * 96)
    header = (
        f"{'scenario':<24} {'strategy':<14} {'steps':>5} {'input_tok':>9} "
        f"{'hist_full~':>10} {'hist_peak~':>10} {'dropped':>7} {'score':>6}"
    )
    print(header)
    lines.append(header)

    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            res = run_one(scenario, strategy)
            line = (
                f"{res['scenario']:<24} {res['strategy']:<14} {res['steps']:>5} "
                f"{res['input_tokens']:>9} "
                f"{res['full_history_est']:>10} {res['bounded_history_est']:>10} "
                f"{res['dropped_exchanges']:>7} {res['score']:>6.3f}"
            )
            print(line)
            lines.append(line)

    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()