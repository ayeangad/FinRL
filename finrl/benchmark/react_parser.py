import json
import re
from typing import Any
from pydantic import BaseModel

from finrl.env.tools import ToolAction

ALLOWED_TOOLS = {
    "get_order",
    "get_quote",
    "get_quotes",
    "get_executions",
    "classify_order",
    "calculate_metrics",
    "submit_report",
}


class ParseResult(BaseModel):
    thought: str = ""
    action_dict: dict[str, Any] | None = None
    tool_action: ToolAction | None = None
    error: str | None = None


def extract_first_json_object(text: str) -> str | None:
    start_idx = text.find("{")
    if start_idx == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
    return None


def parse_react_output(raw_output: str) -> ParseResult:
    if not raw_output or not isinstance(raw_output, str):
        return ParseResult(error="Empty or non-string model output.")

    thought = ""

    # Search for Thought:
    thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", raw_output, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Search for Action: block
    action_match = re.search(r"Action:\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)
    if not action_match:
        return ParseResult(
            thought=thought or raw_output.strip(),
            error="Missing 'Action:' line in output.",
        )

    action_text = action_match.group(1).strip()
    json_str = extract_first_json_object(action_text)

    if not json_str:
        return ParseResult(
            thought=thought,
            error=f"No valid JSON object found in Action text: '{action_text}'",
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as err:
        return ParseResult(
            thought=thought,
            error=f"Malformed Action JSON payload: {err}. Raw text: '{json_str}'",
        )

    if not isinstance(data, dict):
        return ParseResult(
            thought=thought,
            error=f"Action JSON must be an object, got {type(data)}.",
        )

    tool_name = data.get("tool_name")
    if not tool_name or tool_name not in ALLOWED_TOOLS:
        return ParseResult(
            thought=thought,
            action_dict=data,
            error=f"Unknown or missing tool_name '{tool_name}'. Must be one of {sorted(list(ALLOWED_TOOLS))}.",
        )

    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        return ParseResult(
            thought=thought,
            action_dict=data,
            error=f"Tool arguments must be a dictionary, got {type(arguments)}.",
        )

    try:
        tool_action = ToolAction(tool_name=tool_name, arguments=arguments)
        return ParseResult(
            thought=thought,
            action_dict=data,
            tool_action=tool_action,
            error=None,
        )
    except Exception as exc:
        return ParseResult(
            thought=thought,
            action_dict=data,
            error=f"Failed to construct ToolAction: {exc}",
        )
