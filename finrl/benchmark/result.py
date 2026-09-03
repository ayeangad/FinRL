from pydantic import BaseModel, Field


class ScenarioResult(BaseModel):
    scenario_id: str
    model_name: str
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    critical_errors: int = 0
    major_errors: int = 0
    minor_errors: int = 0
    numeric_accuracy: float = Field(ge=0.0, le=1.0)
    regulatory_accuracy: float = Field(ge=0.0, le=1.0)
    evidence_accuracy: float = Field(ge=0.0, le=1.0)
    workflow_integrity: float = Field(ge=0.0, le=1.0)
    tool_calls: int = 0
    steps: int = 0
    invalid_actions: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class BenchmarkResult(BaseModel):
    model_name: str
    total_scenarios: int
    success_rate: float = Field(ge=0.0, le=1.0)
    average_score: float = Field(ge=0.0, le=1.0)
    critical_error_rate: float = Field(ge=0.0)
    major_error_rate: float = Field(ge=0.0)
    minor_error_rate: float = Field(ge=0.0)
    avg_numeric_accuracy: float = Field(ge=0.0, le=1.0)
    avg_regulatory_accuracy: float = Field(ge=0.0, le=1.0)
    avg_evidence_accuracy: float = Field(ge=0.0, le=1.0)
    avg_workflow_integrity: float = Field(ge=0.0, le=1.0)
    avg_tool_calls: float = 0.0
    avg_steps: float = 0.0
    total_invalid_actions: int = 0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    scenario_results: list[ScenarioResult] = Field(default_factory=list)
