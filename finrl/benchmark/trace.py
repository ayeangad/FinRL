from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StepTrace(BaseModel):
    step: int
    observation: dict[str, Any] = Field(default_factory=dict)
    raw_model_output: str
    thought: str = ""
    action: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    prompt_tokens: int = 0
    latency_ms: float = 0.0


class AgentTrace(BaseModel):
    trace_id: str
    schema_version: str = "v1"
    scenario_id: str
    model_name: str
    model_revision: str = "default"
    prompt_version: str = "rule_605_v1"
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    steps: list[StepTrace] = Field(default_factory=list)
    submission: str | None = None
    evaluation: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(
        default_factory=lambda: {
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    )


# Per-1K-token prices (USD). Local models cost ~ electricity; API models billed.
TOKEN_PRICES_USD_PER_1K: dict[str, dict[str, float]] = {
    "qwen_mock": {"input": 0.0, "output": 0.0},
    "qwen_real": {"input": 0.0, "output": 0.0},
    "openai_mock": {"input": 0.0, "output": 0.0},
    "openai_real": {"input": 0.002, "output": 0.008},
    "default": {"input": 0.0, "output": 0.0},
}


def estimate_cost_usd(input_tokens: int, output_tokens: int, model_name: str = "default") -> float:
    key = model_name if model_name in TOKEN_PRICES_USD_PER_1K else "default"
    prices = TOKEN_PRICES_USD_PER_1K[key]
    return round(input_tokens / 1000 * prices["input"] + output_tokens / 1000 * prices["output"], 6)


def save_trace(trace: AgentTrace, base_dir: Path | str = "traces") -> Path:
    out_dir = Path(base_dir) / trace.prompt_version / trace.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{trace.scenario_id}.json"
    filepath.write_text(trace.model_dump_json(indent=2))
    return filepath
