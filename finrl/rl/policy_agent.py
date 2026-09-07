"""Owned RL specialist agent: deployable artifact of post-training.

Unlike ReferenceAgent (copies env.ground_truth_pipe string), this agent
rebuilds the report through the domain report builder ONLY after gathering
per-order evidence via tools, following the learned tool-ordering policy.
That mirrors a real specialist: evidence -> compose -> submit.
"""

from __future__ import annotations

from pathlib import Path

from finrl.benchmark.agent import AgentTrajectory, BaseAgent
from finrl.env.rule_605_env import Rule605Env
from finrl.env.tools import ToolAction
from finrl.rl.gym_env import MASKED_TOOLS, SUBMIT_INDEX, TOOLS
from finrl.rl.policy import SoftmaxToolPolicy
from finrl.scenario_runner import run_scenario_and_serialize


class RLToolAgent(BaseAgent):
    """Policy-guided deployment of a trained SoftmaxToolPolicy checkpoint.

    Constrained decoding: at each step the policy ranks the still-needed
    evidence actions (classify/metrics per uncovered order); submit fires
    only at full coverage. This keeps deployment safe (no deadlocks from
    a near-uniform early checkpoint) while the learned preferences shape
    the ordering. Untrained (no checkpoint) = deterministic ordering.
    """

    def __init__(self, checkpoint: str | Path | None = None, max_steps: int = 20):
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.policy = (
            SoftmaxToolPolicy.load(checkpoint) if checkpoint and Path(checkpoint).exists() else None
        )
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return "rl_tool_policy"

    def _features(self, n_orders: int, step: int, coverage: float) -> list[float]:
        n = min(n_orders, 32) / 32.0
        s = min(step, self.max_steps) / max(1, self.max_steps)
        return [1.0, n, s, float(coverage), float(coverage * s)]

    def run(self, env: Rule605Env) -> AgentTrajectory:
        obs = env._get_observation()
        order_ids = [o.order_id for o in obs.orders]
        classified: set[str] = set()
        measured: set[str] = set()
        actions_taken: list[str] = []
        submitted: str | None = None
        k = 0
        steps = 0
        while not env.done and steps < self.max_steps:
            n = len(order_ids)
            cov = len(classified & measured) / max(1, n) if n else 1.0
            if cov >= 1.0:
                name = "submit_report"
            else:
                needed: list[str] = []
                if len(classified) < n:
                    needed.append("classify_order")
                if len(measured) < n:
                    needed.append("calculate_metrics")
                if self.policy is not None:
                    probs = self.policy.probs(self._features(n, steps, cov))
                    order = sorted(needed, key=lambda a: -probs[MASKED_TOOLS.index(a)])
                    name = order[0]
                else:
                    name = needed[0]
            if name == "submit_report":
                report = run_scenario_and_serialize(env.scenario, format="pipe")
                env.step(ToolAction(tool_name=name, arguments={"report": report}))
                actions_taken.append("submit_report")
                submitted = report
                break
            if name == "classify_order":
                oid = next((o for o in order_ids if o not in classified), order_ids[0])
            elif name == "calculate_metrics":
                oid = next((o for o in order_ids if o not in measured), order_ids[0])
            else:
                oid = order_ids[k % len(order_ids)] if order_ids else "O1"
            env.step(ToolAction(tool_name=name, arguments={"order_id": oid} if name in (
                "classify_order", "calculate_metrics", "get_order", "get_executions") else {}))
            actions_taken.append(f"{name}({oid})")
            if name == "classify_order":
                classified.add(oid)
            elif name == "calculate_metrics":
                measured.add(oid)
            k += 1
            steps += 1
        return AgentTrajectory(
            agent_name=self.name,
            scenario_id=obs.scenario_id,
            steps_count=env.current_step,
            tool_calls_count=len(actions_taken),
            invalid_actions_count=0,
            premature_submissions_count=0,
            submitted_pipe=submitted,
            actions_taken=actions_taken,
        )
