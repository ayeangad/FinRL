"""RLVR/SFT dataset export: the seam for LLM post-training (TRL/GRPO/DPO).

Each golden scenario becomes one RLVR item:
  {prompt, ground_truth_pipe, scenario_id, order_types, n_orders}
A future GRPO run uses dense_report_reward() as the verifiable reward,
exactly as RLVR pipelines (DeepSeek-R1 style) require.
"""

from __future__ import annotations

import json
from pathlib import Path


def export_rlvr_dataset(
    scenarios_dir: str | Path = "scenarios/v0.1/golden",
    out_path: str | Path = "checkpoints/rlvr_dataset.jsonl",
    limit: int | None = None,
) -> dict:
    from finrl.env.rule_605_env import Rule605Env

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
            item = {
                "scenario_id": obs.scenario_id,
                "prompt": (
                    "You are an SEC Rule 605 analyst. Given the orders below, "
                    "call classify_order and calculate_metrics per order, then "
                    f"submit the 24-column pipe report. Orders: "
                    f"{[o.model_dump(mode='json') for o in obs.orders]}"
                ),
                "ground_truth_pipe": env.ground_truth_pipe,
                "n_orders": len(obs.orders),
                "order_types": sorted({str(o.order_type) for o in obs.orders}),
            }
            fh.write(json.dumps(item) + "\n")
            n += 1
    return {"items": n, "path": str(out)}
