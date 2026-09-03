import argparse
from pathlib import Path

from finrl.benchmark.agent import BrokenAgent, Qwen4BAgent, ReferenceAgent
from finrl.benchmark.config import BenchmarkConfig
from finrl.benchmark.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="FinRL Rule 605 Agent Benchmark CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="reference",
        choices=["reference", "broken", "qwen3_4b"],
        help="Agent model to evaluate (reference, broken, qwen3_4b)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="scenarios/v0.1/golden",
        help="Directory path containing scenario .json files",
    )

    args = parser.parse_args()

    if args.model == "reference":
        agent = ReferenceAgent()
    elif args.model == "broken":
        agent = BrokenAgent()
    elif args.model == "qwen3_4b":
        agent = Qwen4BAgent()
    else:
        raise ValueError(f"Unknown model agent: {args.model}")

    config = BenchmarkConfig(scenarios_dir=Path(args.scenarios))
    runner = BenchmarkRunner(config)

    print(f"\nRunning FinRL Rule 605 Benchmark for model '{agent.name}' on {config.scenarios_dir}...")
    result = runner.run_benchmark(agent)

    print("\n" + "=" * 50)
    print("           FINRL RULE 605 BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model Name:       {result.model_name}")
    print(f"Total Scenarios:  {result.total_scenarios}")
    print("-" * 50)
    print(f"Success Rate:     {result.success_rate * 100:.1f}%")
    print(f"Average Score:    {result.average_score:.4f}")
    print("-" * 50)
    print("Errors (Avg per task):")
    print(f"  Critical:       {result.critical_error_rate:.2f}")
    print(f"  Major:          {result.major_error_rate:.2f}")
    print(f"  Minor:          {result.minor_error_rate:.2f}")
    print("-" * 50)
    print("Accuracy Breakdown:")
    print(f"  Numeric Acc:    {result.avg_numeric_accuracy * 100:.1f}%")
    print(f"  Regulatory Acc: {result.avg_regulatory_accuracy * 100:.1f}%")
    print(f"  Evidence Acc:   {result.avg_evidence_accuracy * 100:.1f}%")
    print(f"  Workflow Int:   {result.avg_workflow_integrity * 100:.1f}%")
    print("-" * 50)
    print("Latency & Efficiency:")
    print(f"  P50 Latency:    {result.p50_latency_ms:.2f} ms")
    print(f"  P95 Latency:    {result.p95_latency_ms:.2f} ms")
    print(f"  P99 Latency:    {result.p99_latency_ms:.2f} ms")
    print(f"  Avg Tool Calls: {result.avg_tool_calls:.2f}")
    print(f"  Avg Steps:      {result.avg_steps:.2f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
