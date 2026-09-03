from pathlib import Path

from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    scenarios_dir: Path = Field(
        default_factory=lambda: Path("scenarios/v0.1/golden")
    )
    limit: int | None = None
    max_steps_per_task: int = 50
    critical_error_penalty: float = 0.5
    major_error_penalty: float = 0.2
    minor_error_penalty: float = 0.05
