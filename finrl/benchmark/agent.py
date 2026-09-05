from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from finrl.benchmark.context_selector import ContextSelector
from finrl.benchmark.react_parser import parse_react_output
from finrl.benchmark.trace import AgentTrace, StepTrace, save_trace
from finrl.env.rule_605_env import Rule605Env
from finrl.env.tools import ToolAction
from finrl.models.openai import OpenAIRunner
from finrl.models.qwen3_0_6b import Qwen3_0_6B_Runner


class ConversationWindow:
    """Bounded, recency-aware conversation history for prompt construction.

    Entries are physically pruned once they exceed the token budget, with the
    ``keep_last_n`` most-recent (model, tool) exchange-pairs always retained.
    Token counts are an approximation (~1 token per 4 characters).
    """

    def __init__(self, max_tokens: int = 3000, keep_last_n: int = 3):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.max_tokens = max_tokens
        self.keep_last_n = max(1, keep_last_n)
        self._entries: list[str] = []
        self._entry_tokens: list[int] = []
        self.full_tokens_total = 0
        self.dropped_entries_total = 0
        self.truncation_count = 0

    def append(self, entry: str) -> None:
        estimated = self._estimate_tokens(entry)
        self._entries.append(entry)
        self._entry_tokens.append(estimated)
        self.full_tokens_total += estimated
        self._prune()

    def get_history_str(self) -> str:
        if not self._entries:
            return ""
        body = "\n".join(self._entries)
        if self.dropped_entries_total > 0:
            header = f"[{self.dropped_exchanges} earlier exchanges truncated to fit {self.max_tokens}-token context budget]"
            return f"{header}\n{body}"
        return body

    @property
    def history_tokens_now(self) -> int:
        return sum(self._entry_tokens)

    @property
    def entries_now(self) -> int:
        return len(self._entries)

    @property
    def dropped_exchanges(self) -> int:
        return self.dropped_entries_total // 2

    def _prune(self) -> None:
        protected_count = min(len(self._entries), self.keep_last_n * 2)
        protected_tokens = sum(self._entry_tokens[-protected_count:])
        droppable_end = len(self._entries) - protected_count
        budget = self.max_tokens - protected_tokens

        kept = 0
        used = 0
        for i in range(droppable_end):
            if used + self._entry_tokens[i] > budget:
                break
            used += self._entry_tokens[i]
            kept += 1

        drop_count = droppable_end - kept
        if drop_count <= 0:
            return

        self._entries = self._entries[drop_count:]
        self._entry_tokens = self._entry_tokens[drop_count:]
        self.dropped_entries_total += drop_count
        self.truncation_count += 1

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def _build_context_metrics(
    prompt_strategy: str,
    system_prompt: str,
    window: ConversationWindow,
    selected_sections: list[str],
) -> dict:
    return {
        "prompt_strategy": prompt_strategy,
        "selected_sections": selected_sections,
        "system_prompt_est_tokens": ConversationWindow._estimate_tokens(system_prompt),
        "history_max_tokens": window.max_tokens,
        "history_full_est_tokens": window.full_tokens_total,
        "history_bounded_est_tokens": window.history_tokens_now,
        "dropped_exchanges": window.dropped_exchanges,
        "truncation_events": window.truncation_count,
    }


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
        env.step(act_sub)
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
        env.step(act_sub)

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


def _resolve_context_selector(context_selector: ContextSelector | None, use_scenario_context: bool) -> ContextSelector | None:
    if context_selector is not None:
        return context_selector
    if not use_scenario_context:
        return None
    try:
        return ContextSelector()
    except FileNotFoundError:
        return None


class QwenAgent(BaseAgent):
    def __init__(
        self,
        mode: str = "mock",
        checkpoint: str = "Qwen/Qwen3-0.6B",
        device: str | None = None,
        runner: Qwen3_0_6B_Runner | None = None,
        prompt_path: Path | str = "prompts/rule_605_v1.txt",
        traces_dir: Path | str = "traces",
        context_selector: ContextSelector | None = None,
        use_scenario_context: bool = True,
        max_history_tokens: int = 3000,
        keep_last_n_exchanges: int = 3,
    ):
        self.mode = mode.lower()
        self.checkpoint = checkpoint
        self.device = device
        self.runner = runner or Qwen3_0_6B_Runner(checkpoint=self.checkpoint, device=self.device, mode=self.mode)
        self.prompt_path = Path(prompt_path)
        self.traces_dir = Path(traces_dir)
        self.fallback_prompt = (
            self.prompt_path.read_text()
            if self.prompt_path.exists()
            else "You are an SEC Rule 605 analyst. Return ReAct JSON actions."
        )
        self.context_selector = _resolve_context_selector(context_selector, use_scenario_context)
        self.max_history_tokens = max_history_tokens
        self.keep_last_n_exchanges = keep_last_n_exchanges
        self.prompt_strategy = "scenario_aware" if self.context_selector is not None else "full"

    @property
    def name(self) -> str:
        return f"qwen_{self.mode}"

    def run(self, env: Rule605Env) -> AgentTrajectory:
        obs = env._get_observation()
        started_at = datetime.now(UTC)

        step_traces: list[StepTrace] = []
        window = ConversationWindow(max_tokens=self.max_history_tokens, keep_last_n=self.keep_last_n_exchanges)
        submitted_pipe: str | None = None

        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        invalid_actions_count = 0
        tool_calls_count = 0
        actions_taken = []

        assembled = (
            self.context_selector.build_from_observation(obs.scenario_id, obs.orders)
            if self.context_selector is not None
            else None
        )
        system_prompt = assembled.text if assembled is not None else self.fallback_prompt
        selected_sections = assembled.selected_sections if assembled is not None else []

        while not env.done and env.current_step < env.max_steps:
            current_step_num = env.current_step + 1

            # Format current prompt
            history_str = window.get_history_str()
            prompt = (
                f"{system_prompt}\n\n"
                f"SCENARIO OBJECTIVE:\n"
                f"Scenario ID: {obs.scenario_id}, Security: {obs.security}\n"
                f"Orders: {[o.model_dump(mode='json') for o in obs.orders]}\n\n"
                f"INTERACTION HISTORY:\n{history_str}\n\n"
                f"Step {current_step_num}: Provide your next Thought and Action."
            )

            # Inference call
            raw_output, in_tok, out_tok, lat_ms = self.runner.generate(
                prompt,
                max_new_tokens=256,
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
                window.append(f"Model Output:\n{raw_output}")
                window.append(f"System Error: {parse_res.error}. Please retry with valid ReAct JSON Action.")
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
                break

            window.append(f"Model Output:\n{raw_output}")
            window.append(f"Environment Tool Output:\n{tool_output}")

        completed_at = datetime.now(UTC)

        # Build and save immutable trace
        agent_trace = AgentTrace(
            trace_id=f"{self.name}_{obs.scenario_id}_{int(started_at.timestamp())}",
            schema_version="v1",
            scenario_id=obs.scenario_id,
            model_name=self.name,
            model_revision=self.checkpoint,
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
                "context": _build_context_metrics(self.prompt_strategy, system_prompt, window, selected_sections),
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


class OpenAIAgent(BaseAgent):
    def __init__(
        self,
        mode: str = "mock",
        checkpoint: str = "gpt-5.6-luna",
        device: str | None = None,
        runner: OpenAIRunner | None = None,
        prompt_path: Path | str = "prompts/rule_605_v1.txt",
        traces_dir: Path | str = "traces",
        context_selector: ContextSelector | None = None,
        use_scenario_context: bool = True,
        max_history_tokens: int = 6000,
        keep_last_n_exchanges: int = 3,
    ):
        self.mode = mode.lower()
        self.checkpoint = checkpoint
        self.runner = runner or OpenAIRunner(checkpoint=self.checkpoint, mode=self.mode)
        self.prompt_path = Path(prompt_path)
        self.traces_dir = Path(traces_dir)
        self.fallback_prompt = (
            self.prompt_path.read_text()
            if self.prompt_path.exists()
            else "You are an SEC Rule 605 analyst. Return ReAct JSON actions."
        )
        self.context_selector = _resolve_context_selector(context_selector, use_scenario_context)
        self.max_history_tokens = max_history_tokens
        self.keep_last_n_exchanges = keep_last_n_exchanges
        self.prompt_strategy = "scenario_aware" if self.context_selector is not None else "full"

    @property
    def name(self) -> str:
        return f"openai_{self.mode}"

    def run(self, env: Rule605Env) -> AgentTrajectory:
        obs = env._get_observation()
        started_at = datetime.now(UTC)

        step_traces: list[StepTrace] = []
        window = ConversationWindow(max_tokens=self.max_history_tokens, keep_last_n=self.keep_last_n_exchanges)
        submitted_pipe: str | None = None

        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        invalid_actions_count = 0
        tool_calls_count = 0
        actions_taken = []

        assembled = (
            self.context_selector.build_from_observation(obs.scenario_id, obs.orders)
            if self.context_selector is not None
            else None
        )
        system_prompt = assembled.text if assembled is not None else self.fallback_prompt
        selected_sections = assembled.selected_sections if assembled is not None else []

        while not env.done and env.current_step < env.max_steps:
            current_step_num = env.current_step + 1

            # Format current prompt
            history_str = window.get_history_str()
            prompt = (
                f"{system_prompt}\n\n"
                f"SCENARIO OBJECTIVE:\n"
                f"Scenario ID: {obs.scenario_id}, Security: {obs.security}\n"
                f"Orders: {[o.model_dump(mode='json') for o in obs.orders]}\n\n"
                f"INTERACTION HISTORY:\n{history_str}\n\n"
                f"Step {current_step_num}: Provide your next Thought and Action."
            )

            # Inference call
            raw_output, in_tok, out_tok, lat_ms = self.runner.generate(
                prompt,
                max_new_tokens=2048,
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
                window.append(f"Model Output:\n{raw_output}")
                window.append(f"System Error: {parse_res.error}. Please retry with valid ReAct JSON Action.")
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
                break

            window.append(f"Model Output:\n{raw_output}")
            window.append(f"Environment Tool Output:\n{tool_output}")

        completed_at = datetime.now(UTC)

        # Build and save immutable trace
        agent_trace = AgentTrace(
            trace_id=f"{self.name}_{obs.scenario_id}_{int(started_at.timestamp())}",
            schema_version="v1",
            scenario_id=obs.scenario_id,
            model_name=self.name,
            model_revision=self.checkpoint,
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
                "context": _build_context_metrics(self.prompt_strategy, system_prompt, window, selected_sections),
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
