from pathlib import Path

from pydantic import BaseModel, Field


class OperatorRelease(BaseModel):
    """Eval-gated owned-operator lineage (Hyde-style packaging)."""

    operator_id: str = "rule605-specialist-v0"
    model_revision: str = "default"
    dataset_version: str = "golden-v0.1"
    training_config: dict = Field(default_factory=dict)
    min_success_rate: float = 0.0
    min_average_score: float = 0.0

    def gate_passed(self, success_rate: float, average_score: float) -> bool:
        return success_rate >= self.min_success_rate and average_score >= self.min_average_score


class BenchmarkConfig(BaseModel):
    scenarios_dir: Path = Field(
        default_factory=lambda: Path("scenarios/v0.1/golden")
    )
    limit: int | None = None
    max_steps_per_task: int = 50
    critical_error_penalty: float = 0.5
    major_error_penalty: float = 0.2
    minor_error_penalty: float = 0.05
    # Owned-operator registry (post-training lineage).
    operator: OperatorRelease = Field(default_factory=OperatorRelease)
    dataset_version: str = "golden-v0.1"
