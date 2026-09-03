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
        "--mode",
        type=str,
        default="mock",
        choices=["real", "mock"],
        help="Execution mode: 'real' (loads PyTorch model weights) or 'mock' (deterministic test agent)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Qwen/Qwen3-4B-Instruct",
        help="HuggingFace model ID or local path to model checkpoint",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="scenarios/v0.1/golden",
        help="Directory path containing scenario .json files",
    )

    args = parser.parse_args()

    device_str = "N/A"
    precision_str = "N/A"
    weights_str = "NONE (mock)"

    if args.model == "reference":
        agent = ReferenceAgent()
        weights_str = "NONE (reference logic)"
    elif args.model == "broken":
        agent = BrokenAgent()
        weights_str = "NONE (broken logic)"
    elif args.model == "qwen3_4b":
        agent = Qwen4BAgent(mode=args.mode, checkpoint=args.checkpoint)
        device_str = agent.runner.device or "cpu"
        precision_str = agent.runner.precision_str
        weights_str = agent.runner.checkpoint if args.mode == "real" else "NONE (mock)"
    else:
        raise ValueError(f"Unknown model agent: {args.model}")

    config = BenchmarkConfig(scenarios_dir=Path(args.scenarios))
    runner = BenchmarkRunner(config)

    print("\n" + "=" * 60)
    print("           FINRL RULE 605 BENCHMARK")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Mode:        {args.mode.upper()}")
    print(f"Device:      {device_str}")
    print(f"Precision:   {precision_str}")
    print(f"Weights:     {weights_str}")
    print(f"Scenarios:   100")
    print(f"Prompt:      rule_605_v1")
    print("=" * 60 + "\n")

    result = runner.run_benchmark(agent)

    print("-" * 60)
    print("           FINRL RULE 605 BENCHMARK RESULTS")
    print("-" * 60)
    print(f"Model Name:       {result.model_name}")
    print(f"Total Scenarios:  {result.total_scenarios}")
    print("-" * 60)
    print(f"Success Rate:     {result.success_rate * 100:.1f}%")
    print(f"Average Score:    {result.average_score:.4f}")
    print("-" * 60)
    print("Errors (Avg per task):")
    print(f"  Critical:       {result.critical_error_rate:.2f}")
    print(f"  Major:          {result.major_error_rate:.2f}")
    print(f"  Minor:          {result.minor_error_rate:.2f}")
    print("-" * 60)
    print("Accuracy Breakdown:")
    print(f"  Numeric Acc:    {result.avg_numeric_accuracy * 100:.1f}%")
    print(f"  Regulatory Acc: {result.avg_regulatory_accuracy * 100:.1f}%")
    print(f"  Evidence Acc:   {result.avg_evidence_accuracy * 100:.1f}%")
    print(f"  Workflow Int:   {result.avg_workflow_integrity * 100:.1f}%")
    print("-" * 60)
    print("Latency & Efficiency:")
    print(f"  P50 Latency:    {result.p50_latency_ms:.2f} ms")
    print(f"  P95 Latency:    {result.p95_latency_ms:.2f} ms")
    print(f"  P99 Latency:    {result.p99_latency_ms:.2f} ms")
    print(f"  Avg Tool Calls: {result.avg_tool_calls:.2f}")
    print(f"  Avg Steps:      {result.avg_steps:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
