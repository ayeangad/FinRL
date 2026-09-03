from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from finrl.benchmark.react_parser import parse_react_output
from finrl.benchmark.trace import AgentTrace, StepTrace, save_trace
from finrl.env.rule_605_env import Rule605Env
from finrl.env.tools import ToolAction
from finrl.models.qwen3_4b import Qwen3_4B_Runner


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


class Qwen4BAgent(BaseAgent):
    def __init__(
        self,
        runner: Qwen3_4B_Runner | None = None,
        prompt_path: Path | str = "prompts/rule_605_v1.txt",
        traces_dir: Path | str = "traces",
    ):
        self.runner = runner or Qwen3_4B_Runner()
        self.prompt_path = Path(prompt_path)
        self.traces_dir = Path(traces_dir)
        self.system_prompt = (
            self.prompt_path.read_text()
            if self.prompt_path.exists()
            else "You are an SEC Rule 605 analyst. Return ReAct JSON actions."
        )

    @property
    def name(self) -> str:
        return "qwen3_4b"

    def run(self, env: Rule605Env) -> AgentTrajectory:
        obs = env._get_observation()
        started_at = datetime.now(timezone.utc)

        step_traces: list[StepTrace] = []
        conversation_history: list[str] = []
        submitted_pipe: str | None = None

        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        invalid_actions_count = 0
        tool_calls_count = 0
        actions_taken = []

        while not env.done and env.current_step < env.max_steps:
            current_step_num = env.current_step + 1

            # Format current prompt
            history_str = "\n".join(conversation_history)
            prompt = (
                f"{self.system_prompt}\n\n"
                f"SCENARIO OBJECTIVE:\n"
                f"Scenario ID: {obs.scenario_id}, Security: {obs.security}\n"
                f"Orders: {[o.model_dump(mode='json') for o in obs.orders]}\n\n"
                f"INTERACTION HISTORY:\n{history_str}\n\n"
                f"Step {current_step_num}: Provide your next Thought and Action."
            )

            # Inference call
            raw_output, in_tok, out_tok, lat_ms = self.runner.generate(
                prompt,
                max_new_tokens=512,
                temperature=0.0,
            )

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_latency_ms += lat_ms

            # Strict ReAct parsing
            parse_res = parse_react_output(raw_output)

            step_trace = StepTrace(
                step=current_step_num,
                observation=obs.model_dump(mode="json"),
                raw_model_output=raw_output,
                thought=parse_res.thought,
                action=parse_res.action_dict,
                latency_ms=lat_ms,
            )

            if parse_res.error or not parse_res.tool_action:
                invalid_actions_count += 1
                conversation_history.append(f"Model Output:\n{raw_output}")
                conversation_history.append(f"System Error: {parse_res.error}. Please retry with valid ReAct JSON Action.")
                step_traces.append(step_trace)
                continue

            tool_action = parse_res.tool_action
            tool_calls_count += 1
            actions_taken.append(f"{tool_action.tool_name}({tool_action.arguments})")

            # Environment Step
            step_res = env.step(tool_action)
            obs = step_res.observation
            tool_output = step_res.info.get("tool_output")

            step_trace.tool_result = {"output": tool_output, "reward": step_res.reward}
            step_traces.append(step_trace)

            if tool_action.tool_name == "submit_report":
                submitted_pipe = tool_action.arguments.get("report")

            conversation_history.append(f"Model Output:\n{raw_output}")
            conversation_history.append(f"Environment Tool Output:\n{tool_output}")

        completed_at = datetime.now(timezone.utc)

        # Build and save immutable trace
        agent_trace = AgentTrace(
            trace_id=f"{self.name}_{obs.scenario_id}_{int(started_at.timestamp())}",
            schema_version="v1",
            scenario_id=obs.scenario_id,
            model_name=self.name,
            model_revision="qwen2.5-3b-instruct",
            prompt_version="rule_605_v1",
            started_at=started_at,
            completed_at=completed_at,
            steps=step_traces,
            submission=submitted_pipe,
            evaluation={"ground_truth_pipe": env.ground_truth_pipe},
            metrics={
                "latency_ms": round(total_latency_ms, 2),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "cost_usd": 0.0,
            },
        )
        save_trace(agent_trace, base_dir=self.traces_dir)

        return AgentTrajectory(
            agent_name=self.name,
            scenario_id=obs.scenario_id,
            steps_count=env.current_step,
            tool_calls_count=tool_calls_count,
            invalid_actions_count=invalid_actions_count,
            premature_submissions_count=0,
            submitted_pipe=submitted_pipe,
            actions_taken=actions_taken,
        )
