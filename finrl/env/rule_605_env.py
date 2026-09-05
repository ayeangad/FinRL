from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from finrl.env.state import (
    EnvObservation,
    ObservableOrder,
)
from finrl.env.tools import ToolAction
from finrl.evals.order_evaluator import evaluate_order
from finrl.scenario import Scenario
from finrl.scenario_runner import (
    load_scenario,
    parse_scenario,
    run_scenario_and_serialize,
)


class StepResult(BaseModel):
    observation: EnvObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class Rule605Env:
    def __init__(self, max_steps: int = 50, max_action_history: int = 20):
        self.max_steps = max_steps
        self.max_action_history = max_action_history
        self.scenario: Scenario | None = None
        self.current_step: int = 0
        self.done: bool = False
        self.action_history: list[str] = []
        self.ground_truth_report: Any = None
        self.ground_truth_pipe: str = ""

    def reset(self, scenario_or_path: Scenario | str | Path) -> EnvObservation:
        if isinstance(scenario_or_path, (str, Path)):
            self.scenario = load_scenario(scenario_or_path)
        else:
            self.scenario = scenario_or_path

        self.current_step = 0
        self.done = False
        self.action_history = []

        # Parse ground truth
        self.ground_truth_pipe = run_scenario_and_serialize(self.scenario, format="pipe")

        return self._get_observation()

    def _get_observation(self) -> EnvObservation:
        assert self.scenario is not None
        obs_orders = [
            ObservableOrder(
                order_id=o.order_id,
                security=self.scenario.security,
                side=o.side,
                order_type=o.order_type,
                quantity=o.quantity,
                limit_price=o.limit_price,
                stop_price=o.stop_price,
                received_at=o.received_at,
            )
            for o in self.scenario.get_all_orders()
        ]

        return EnvObservation(
            scenario_id=self.scenario.scenario_id,
            security=self.scenario.security,
            orders=obs_orders,
            quotes_count=len(self.scenario.quotes),
            executions_count=len(self.scenario.executions),
            current_step=self.current_step,
            action_history=list(self.action_history),
        )

    def step(self, action: ToolAction) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.current_step += 1
        self.action_history.append(f"{action.tool_name}({action.arguments})")
        if self.max_action_history and len(self.action_history) > self.max_action_history:
            self.action_history = self.action_history[-self.max_action_history :]

        tool_output: Any = None
        reward: float = 0.0
        info: dict[str, Any] = {}

        if action.tool_name == "get_order":
            order_id = action.arguments.get("order_id")
            orders = self.scenario.get_all_orders()
            match = next((o for o in orders if o.order_id == order_id), None)
            tool_output = match.model_dump(mode="json") if match else None

        elif action.tool_name == "get_quote":
            ts_str = action.arguments.get("timestamp")
            ts = datetime.fromisoformat(ts_str) if ts_str else None
            domain_orders, market, _ = parse_scenario(self.scenario)
            quote = market.quote_at(ts) if ts else None
            tool_output = quote.model_dump(mode="json") if quote else None

        elif action.tool_name == "get_quotes":
            start_str = action.arguments.get("start_time")
            end_str = action.arguments.get("end_time")
            start = datetime.fromisoformat(start_str) if start_str else datetime.min.replace(tzinfo=UTC)
            end = datetime.fromisoformat(end_str) if end_str else datetime.max.replace(tzinfo=UTC)
            matching = [
                q.model_dump(mode="json")
                for q in self.scenario.quotes
                if start <= q.timestamp <= end
            ]
            tool_output = matching

        elif action.tool_name == "get_executions":
            order_id = action.arguments.get("order_id")
            matching = [
                e.model_dump(mode="json")
                for e in self.scenario.executions
                if e.order_id == order_id
            ]
            tool_output = matching

        elif action.tool_name == "classify_order":
            order_id = action.arguments.get("order_id")
            domain_orders, market, domain_execs = parse_scenario(self.scenario)
            target = next((o for o in domain_orders if o.order_id == order_id), None)
            if target:
                target_execs = [e for e in domain_execs if e.order_id == order_id]
                report = evaluate_order(target, target_execs, market)
                tool_output = {
                    "order_id": order_id,
                    "order_type_category": report.order_type_category.value,
                    "order_size_bucket": report.order_size_bucket.value,
                    "reportable": report.reportable,
                }
            else:
                tool_output = None

        elif action.tool_name == "calculate_metrics":
            order_id = action.arguments.get("order_id")
            domain_orders, market, domain_execs = parse_scenario(self.scenario)
            target = next((o for o in domain_orders if o.order_id == order_id), None)
            if target:
                target_execs = [e for e in domain_execs if e.order_id == order_id]
                report = evaluate_order(target, target_execs, market)
                tool_output = report.model_dump(mode="json")
            else:
                tool_output = None

        elif action.tool_name == "submit_report":
            submitted_pipe = action.arguments.get("report")
            self.done = True

            if submitted_pipe == self.ground_truth_pipe:
                reward = 1.0
                info["exact_match"] = True
            else:
                sub_lines = submitted_pipe.strip().split("\n") if submitted_pipe else []
                gt_lines = self.ground_truth_pipe.strip().split("\n")
                matching_count = sum(1 for s, g in zip(sub_lines, gt_lines) if s == g)
                reward = round(matching_count / len(gt_lines), 4)
                info["exact_match"] = False
                info["matching_lines"] = matching_count
                info["total_lines"] = len(gt_lines)

            tool_output = {"status": "submitted", "reward": reward}

        if self.current_step >= self.max_steps and not self.done:
            self.done = True
            info["truncated"] = True

        info["tool_output"] = tool_output
        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info=info,
        )
