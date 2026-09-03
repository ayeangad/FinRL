from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolAction(BaseModel):
    tool_name: Literal[
        "get_order",
        "get_quote",
        "get_quotes",
        "get_executions",
        "classify_order",
        "calculate_metrics",
        "submit_report",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)
