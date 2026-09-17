"""RLVR/SFT dataset export: the seam for LLM post-training (TRL/GRPO/DPO).

Each scenario becomes one RLVR item:
  {prompt, ground_truth_pipe, scenario_id, order_types, n_orders,
   split, dataset_version, prompt_version}
A GRPO run uses dense_report_reward() as the verifiable reward, exactly
as RLVR pipelines (DeepSeek-R1 style) require.

Ground-truth hygiene: the prompt contains orders/quotes metadata ONLY.
The pipe report lives in a separate field (the reward target), never in
the prompt. Default export is train-split only to prevent test leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET_VERSION = "rlvr-v1"
PROMPT_VERSION = "rule605-rlvr-v1"


def export_rlvr_dataset(
    scenarios_dir: str | Path = "scenarios/v0.1/golden",
    out_path: str | Path = "checkpoints/rlvr_dataset.jsonl",
    limit: int | None = None,
    split: str | None = "train",
) -> dict:
    from finrl.env.rule_605_env import Rule605Env

    if split in ("train", "val", "test"):
        try:
            from finrl.rl.splits import get_split

            files = [Path(p) for p in get_split(split, scenarios_dir)]
        except Exception:
            files = sorted(Path(scenarios_dir).glob("*.json"))
    else:
        files = sorted(Path(scenarios_dir).glob("*.json"))
    if limit is not None:
        files = files[:limit]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for f in files:
            env = Rule605Env()
            obs = env.reset(f)
            orders_json = [o.model_dump(mode="json") for o in obs.orders]
            item = {
                "scenario_id": obs.scenario_id,
                "prompt": (
                    "You are an SEC Rule 605 analyst. Given the orders below, "
                    "call classify_order and calculate_metrics per order, then "
                    f"submit the 24-column pipe report. Orders: {orders_json}"
                ),
                "ground_truth_pipe": env.ground_truth_pipe,
                "n_orders": len(obs.orders),
                "order_types": sorted({str(o.order_type) for o in obs.orders}),
                "split": split,
                "dataset_version": DATASET_VERSION,
                "prompt_version": PROMPT_VERSION,
            }
            fh.write(json.dumps(item) + "\n")
            n += 1
    return {"items": n, "path": str(out), "split": split, "dataset_version": DATASET_VERSION}
