from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

from finrl.env.rule_605_env import Rule605Env
from finrl.env.tools import ToolAction


class AgentTrajectory(BaseModel):
    agent_name: str
    scenario_id: str
    steps_count: int = 0
    tool_calls_count: int = 0
    invalid_actions_count: int = 0
    premature_submissions_count: int = 0
    submitted_pipe: str | None = None
    actions_taken: list[str] = Field(default_factory=list)


class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, env: Rule605Env) -> AgentTrajectory:
        pass


class ReferenceAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "reference"

    def run(self, env: Rule605Env) -> AgentTrajectory:
        obs = env._get_observation()
        actions = []

        # 1. Inspect orders
        for order in obs.orders:
            act_cls = ToolAction(tool_name="classify_order", arguments={"order_id": order.order_id})
            env.step(act_cls)
            actions.append(f"classify_order({order.order_id})")

            act_met = ToolAction(tool_name="calculate_metrics", arguments={"order_id": order.order_id})
            env.step(act_met)
            actions.append(f"calculate_metrics({order.order_id})")

        # 2. Submit ground truth pipe report
        gt_pipe = env.ground_truth_pipe
        act_sub = ToolAction(tool_name="submit_report", arguments={"report": gt_pipe})
        res = env.step(act_sub)
        actions.append("submit_report")

        return AgentTrajectory(
            agent_name=self.name,
            scenario_id=obs.scenario_id,
            steps_count=env.current_step,
            tool_calls_count=env.current_step,
            invalid_actions_count=0,
            premature_submissions_count=0,
            submitted_pipe=gt_pipe,
            actions_taken=actions,
        )


class BrokenAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "broken"

    def run(self, env: Rule605Env) -> AgentTrajectory:
        obs = env._get_observation()
        # Submit malformed pipe report immediately
        bad_pipe = "order_type_category|order_size_bucket|num_covered_orders\nmarket|100_to_499|999"
        act_sub = ToolAction(tool_name="submit_report", arguments={"report": bad_pipe})
        res = env.step(act_sub)

        return AgentTrajectory(
            agent_name=self.name,
            scenario_id=obs.scenario_id,
            steps_count=1,
            tool_calls_count=1,
            invalid_actions_count=0,
            premature_submissions_count=1,
            submitted_pipe=bad_pipe,
            actions_taken=["submit_report(malformed)"],
        )
