"""GRPO / RLVR stub for LLM post-training (TRL track).

Verifiable reward = dense_report_reward (same fn the classic RL track
uses for PPO). Dataset = checkpoints/rlvr_dataset.jsonl (train split).

- If `trl` + a causal LM are available, `train_grpo()` launches a real
  GRPOTrainer run.
- Otherwise `smoke()` runs a CPU-only demonstration: score reference vs
  broken submissions on the first N items to prove the reward seam works
  end-to-end without GPU weights.

Usage:
  python -m finrl.rl.train_grpo_stub --smoke --limit 10
  python -m finrl.rl.train_grpo_stub --model Qwen/Qwen2.5-0.5B --output-dir checkpoints/grpo_rule605
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reward_for_pair(submitted: str | None, ground_truth: str) -> dict:
    from finrl.rl.reward import dense_report_reward

    return dense_report_reward(submitted, ground_truth)


def smoke(dataset: str | Path = "checkpoints/rlvr_dataset.jsonl", limit: int = 10) -> dict:
    from finrl.rl.reward import dense_report_reward

    path = Path(dataset)
    if not path.exists():
        from finrl.rl.dataset import export_rlvr_dataset

        export_rlvr_dataset(out_path=path, split="train", limit=50)
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    ref_scores, broken_scores = [], []
    for item in rows:
        gt = item["ground_truth_pipe"]
        ref_scores.append(dense_report_reward(gt, gt)["reward"])
        broken_scores.append(dense_report_reward("invalid_pipe_data", gt)["reward"])
    import numpy as np

    return {
        "mode": "smoke",
        "items": len(rows),
        "ref_mean": round(float(np.mean(ref_scores)) if ref_scores else 0.0, 4),
        "broken_mean": round(float(np.mean(broken_scores)) if broken_scores else 0.0, 4),
        "reward_fn": "finrl.rl.reward.dense_report_reward",
        "dataset": str(path),
    }


def train_grpo(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    dataset_path: str | Path = "checkpoints/rlvr_dataset.jsonl",
    output_dir: str | Path = "checkpoints/grpo_rule605",
    max_steps: int = 100,
) -> dict:
    """Launch TRL GRPOTrainer if available, else raise with instructions."""
    try:
        import trl  # noqa: F401
        from datasets import load_dataset  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "GRPO training needs `trl`, `datasets`, `transformers` (+ torch). "
            f"Import failed: {e}. Run the smoke path instead: "
            "python -m finrl.rl.train_grpo_stub --smoke"
        ) from e
    # Import here so module import stays lightweight without GPU deps.
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    from finrl.rl.reward import dense_report_reward

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    def _reward(prompts, completions, ground_truth=None, **kwargs):
        scores = []
        for comp, gt in zip(completions, ground_truth or []):
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            scores.append(dense_report_reward(text, gt)["reward"])
        return scores

    args = GRPOConfig(output_dir=str(output_dir), max_steps=max_steps)
    trainer = GRPOTrainer(model=model_name, args=args, train_dataset=ds, reward_funcs=_reward)
    trainer.train()
    trainer.save_model(str(output_dir))
    return {"mode": "grpo", "model": model_name, "output_dir": str(output_dir), "steps": max_steps}


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO/RLVR stub for Rule 605")
    ap.add_argument("--smoke", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--dataset", default="checkpoints/rlvr_dataset.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--output-dir", default="checkpoints/grpo_rule605")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--train", action="store_true", help="Attempt real TRL GRPO run")
    args = ap.parse_args()
    if args.train:
        print(json.dumps(train_grpo(args.model, args.dataset, args.output_dir, args.max_steps), indent=2))
    else:
        print(json.dumps(smoke(args.dataset, args.limit), indent=2))


if __name__ == "__main__":
    main()
