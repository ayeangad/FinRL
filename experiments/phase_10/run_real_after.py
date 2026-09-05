"""Controlled real Qwen3-0.6B experiment on representative scenarios (phase-10).

Runs the phase-10 "after" system (scenario-aware sections + windowed history)
in real inference mode. Each scenario runs in an isolated subprocess with a
wall-clock cap, so a degenerate loop (e.g. stop_01 in phase 9) is recorded as
termination="timeout" instead of hanging the batch for 35+ minutes.

Captures per run: scenario type, steps, total input tokens, peak prompt tokens
(max per-step real prompt length), truncation events, dropped exchanges,
score, runtime, and termination reason.

Batch:   .venv/bin/python -m experiments.phase_10.run_real_after
Single:  .venv/bin/python -m experiments.phase_10.run_real_after --run-one stop_01.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from finrl.benchmark.agent import QwenAgent
from finrl.benchmark.evaluator import evaluate_submission
from finrl.env.rule_605_env import Rule605Env

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "scenarios" / "v0.1" / "golden"
TRACES_ROOT = REPO / "traces" / "phase_10"
RUN_RESULTS = TRACES_ROOT / "run_results"

STRATEGY = "scenario_aware"
MAX_STEPS = 30
TIMEOUT_S = 600

SCENARIOS = [
    "market_01.json",
    "marketable_limit_01.json",
    "midpoint_limit_01.json",
    "multi_exec_01.json",
    "stop_01.json",
]

SCENARIO_TYPES = {
    "market_01.json": "market",
    "marketable_limit_01.json": "limit",
    "midpoint_limit_01.json": "limit",
    "multi_exec_01.json": "multi",
    "stop_01.json": "stop",
}

RESULT_PATH = {
    "market_01.json": RUN_RESULTS / "market_01.json",
    "marketable_limit_01.json": RUN_RESULTS / "marketable_limit_01.json",
    "midpoint_limit_01.json": RUN_RESULTS / "midpoint_limit_01.json",
    "multi_exec_01.json": RUN_RESULTS / "multi_exec_01.json",
    "stop_01.json": RUN_RESULTS / "stop_01.json",
}


def run_one(scenario: str, max_steps: int) -> dict:
    env = Rule605Env(max_steps=max_steps)
    env.reset(GOLDEN / scenario)

    traces_dir = TRACES_ROOT / STRATEGY
    agent = QwenAgent(
        mode="real",
        traces_dir=traces_dir,
        use_scenario_context=True,
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
    steps = trace["steps"]
    ctx = trace.get("metrics", {}).get("context", {})

    return {
        "scenario": scenario,
        "strategy": STRATEGY,
        "termination": trace.get("metrics", {}).get("termination", "unknown"),
        "steps": traj.steps_count,
        "invalid_actions": traj.invalid_actions_count,
        "submitted": traj.submitted_pipe is not None,
        "score": eval_detail.score,
        "elapsed_s": round(elapsed, 1),
        "input_tokens": trace.get("metrics", {}).get("input_tokens", 0),
        "output_tokens": trace.get("metrics", {}).get("output_tokens", 0),
        "peak_prompt_tokens": max((s.get("prompt_tokens", 0) for s in steps), default=0),
        "full_history_est": ctx.get("history_full_est_tokens", 0),
        "bounded_history_est": ctx.get("history_bounded_est_tokens", 0),
        "dropped_exchanges": ctx.get("dropped_exchanges", 0),
        "truncation_events": ctx.get("truncation_events", 0),
        "sections": ctx.get("selected_sections", []),
    }


def format_row(res: dict) -> str:
    return (
        f"{res['scenario']:<24} {SCENARIO_TYPES.get(res['scenario'], '?'):<7} "
        f"{res['termination']:<10} {res['steps']:>5} {res['input_tokens']:>9} "
        f"{res['peak_prompt_tokens']:>9} {res['truncation_events']:>6} "
        f"{res['dropped_exchanges']:>8} {res['elapsed_s']:>8} {res['score']:>6.3f}"
    )


def batch(max_steps: int, timeout_s: int) -> None:
    RUN_RESULTS.mkdir(parents=True, exist_ok=True)
    out = REPO / "experiments" / "phase_10" / "real_5scenario.txt"
    lines = []

    print("=" * 104)
    print("REAL Qwen3-0.6B 'after': scenario-aware sections + windowed history (phase-10)")
    print("=" * 104)
    header = (
        f"{'scenario':<24} {'type':<7} {'termination':<10} {'steps':>5} {'input_tok':>9} "
        f"{'peak_prompt':>9} {'trunc':>6} {'dropped':>8} {'elapsed_s':>8} {'score':>6}"
    )
    print(header)
    lines.append(header)

    for scenario in SCENARIOS:
        res_path = RESULT_PATH[scenario]
        if res_path.exists():
            res_path.unlink()
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.phase_10.run_real_after",
                    "--run-one",
                    scenario,
                    "--max-steps",
                    str(max_steps),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            res = {
                "scenario": scenario,
                "strategy": STRATEGY,
                "termination": "timeout",
                "steps": -1,
                "invalid_actions": 0,
                "submitted": False,
                "score": 0.0,
                "elapsed_s": timeout_s,
                "input_tokens": 0,
                "output_tokens": 0,
                "peak_prompt_tokens": 0,
                "full_history_est": 0,
                "bounded_history_est": 0,
                "dropped_exchanges": 0,
                "truncation_events": 0,
                "sections": [],
            }
        else:
            if proc.returncode != 0:
                res = {
                    "scenario": scenario,
                    "strategy": STRATEGY,
                    "termination": "error",
                    "steps": -1,
                    "invalid_actions": 0,
                    "submitted": False,
                    "score": 0.0,
                    "elapsed_s": -1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "peak_prompt_tokens": 0,
                    "full_history_est": 0,
                    "bounded_history_est": 0,
                    "dropped_exchanges": 0,
                    "truncation_events": 0,
                    "sections": [],
                }
            else:
                res = json.loads(res_path.read_text())

        line = format_row(res)
        print(line)
        lines.append(line)

    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport written: {out}")


def single(scenario: str, max_steps: int) -> None:
    res = run_one(scenario, max_steps)
    RUN_RESULTS.mkdir(parents=True, exist_ok=True)
    RESULT_PATH[scenario].write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-one", choices=SCENARIOS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    args = parser.parse_args()

    if args.run_one:
        single(args.run_one, args.max_steps)
    else:
        batch(args.max_steps, args.timeout)


if __name__ == "__main__":
    main()