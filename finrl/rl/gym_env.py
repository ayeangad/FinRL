"""Gymnasium-compatible wrapper around Rule605Env with shaped rewards.

MDP definition:
  state: scenario orders + step + covered tool-use (which order_ids have been
         classified / metric-computed) + action history length.
  action: discrete tool choice in {0..6} + order_id argument binding.
          TOOLS order defines the action space.
  reward: small intermediate shaping per step (+0.05 first classify/metrics
          per order, -0.01 redundant, -0.05 invalid/unknown order, -0.005
          step cost) + dense terminal reward from dense_report_reward on
          submit, or 0 with truncation penalty at max steps.

Compatible with gymnasium if installed (action_space/observation_space
exposed), otherwise usable standalone via reset()/step() tuple API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finrl.env.rule_605_env import Rule605Env
from finrl.env.tools import ToolAction
from finrl.rl.reward import dense_report_reward
from finrl.scenario_runner import run_scenario_and_serialize

TOOLS = [
    "get_order",
    "get_quote",
    "get_quotes",
    "get_executions",
    "classify_order",
    "calculate_metrics",
    "submit_report",
]
SUBMIT_INDEX = TOOLS.index("submit_report")

# Decision-relevant action subset for the control policy. Info tools
# (get_order/get_quote/get_quotes/get_executions) are essential for LLM
# exploration but carry no reward signal, so the default RL action space
# masks to the three actions that affect return. Full 7-action space
# remains available via action_set="full".
MASKED_TOOLS = ["classify_order", "calculate_metrics", "submit_report"]

FIRST_EVIDENCE_BONUS = 0.05
REDUNDANT_PENALTY = -0.01
INVALID_PENALTY = -0.05
STEP_COST = -0.005


class Rule605GymEnv:
    """Thin learnable wrapper. The underlying env stays the source of truth."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 50, action_set: str = "masked"):
        self.gym_max_steps = max_steps
        if action_set not in ("masked", "full"):
            raise ValueError("action_set must be 'masked' or 'full'")
        self.action_set = action_set
        self.action_names = MASKED_TOOLS if action_set == "masked" else TOOLS
        self.env = Rule605Env(max_steps=max_steps)
        self.classified: set[str] = set()
        self.measured: set[str] = set()
        self._order_ids: list[str] = []
        self.action_space_n = len(self.action_names)
        try:
            import gymnasium as gym  # type: ignore

            self.action_space = gym.spaces.Discrete(len(self.action_names))
            self.observation_space = gym.spaces.Dict(
                {
                    "n_orders": gym.spaces.Box(low=0, high=10_000, shape=(1,), dtype=int),
                    "step": gym.spaces.Box(low=0, high=10_000, shape=(1,), dtype=int),
                    "coverage": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
                }
            )
        except Exception:
            self.action_space = None
            self.observation_space = None

    # -- core API ---------------------------------------------------------
    def reset(self, scenario_or_path=None, seed: int | None = None, options: dict | None = None):
        if scenario_or_path is None and options and "scenario" in options:
            scenario_or_path = options["scenario"]
        if scenario_or_path is None:
            raise ValueError("reset() requires a scenario path or Scenario")
        obs = self.env.reset(scenario_or_path)
        self.classified = set()
        self.measured = set()
        self._order_ids = [o.order_id for o in self.env.scenario.get_all_orders()]
        out = self._obs_dict()
        info = {"scenario_id": obs.scenario_id, "n_orders": len(self._order_ids)}
        return (out, info) if self.observation_space is not None else out

    def _obs_dict(self) -> dict[str, Any]:
        n = len(self._order_ids)
        covered = len(self.classified & self.measured)
        return {
            "n_orders": n,
            "step": self.env.current_step,
            "coverage": (covered / max(1, n)),
            "classified": sorted(self.classified),
            "measured": sorted(self.measured),
            "done": self.env.done,
        }

    def observation_features(self) -> list[float]:
        """Fixed 5-dim feature vector for the linear policy."""
        o = self._obs_dict()
        n = min(o["n_orders"], 32) / 32.0
        step = min(o["step"], self.gym_max_steps) / max(1, self.gym_max_steps)
        return [1.0, n, step, float(o["coverage"]), float(o["coverage"] * step)]

    def step(self, action: int | dict | ToolAction, order_id: str | None = None):
        tool_action = self._coerce_action(action, order_id)
        shaped = STEP_COST
        if tool_action.tool_name in ("classify_order", "calculate_metrics"):
            oid = tool_action.arguments.get("order_id")
            if oid not in self._order_ids:
                shaped += INVALID_PENALTY
            else:
                store = self.classified if tool_action.tool_name == "classify_order" else self.measured
                shaped += FIRST_EVIDENCE_BONUS if oid not in store else REDUNDANT_PENALTY
                store.add(oid)
        elif tool_action.tool_name == "submit_report":
            pass  # terminal reward below replaces shaping
        else:
            shaped += 0.0  # info tools: neutral + step cost only

        result = self.env.step(tool_action)
        terminated = result.done and "truncated" not in result.info
        truncated = bool(result.info.get("truncated", False))

        if tool_action.tool_name == "submit_report":
            dense = dense_report_reward(
                tool_action.arguments.get("report"), self.env.ground_truth_pipe
            )
            reward = float(dense["reward"])
            result.info["dense"] = dense
            result.info["shaped_pretotal"] = shaped
        elif truncated:
            reward = -0.1  # failed to submit within budget
            result.info["dense"] = {"reward": 0.0}
        else:
            reward = float(shaped)
            result.reward = reward

        obs = self._obs_dict()
        if self.observation_space is not None:
            return obs, reward, terminated, truncated, result.info
        return obs, reward, result.done, result.info

    # -- helpers ----------------------------------------------------------
    def _coerce_action(self, action, order_id=None) -> ToolAction:
        if isinstance(action, ToolAction):
            return action
        if isinstance(action, dict):
            return ToolAction(
                tool_name=action.get("tool_name", "get_order"),
                arguments=action.get("arguments", {}),
            )
        idx = int(action)
        if self.action_set == "masked":
            name = self.action_names[idx % len(self.action_names)]
            if name == "submit_report":
                report = self._build_evidence_report()
                return ToolAction(tool_name=name, arguments={"report": report})
            oid = order_id or (self._order_ids[0] if self._order_ids else "O1")
            return ToolAction(tool_name=name, arguments={"order_id": oid})
        name = TOOLS[idx % len(TOOLS)]
        if name == "submit_report":
            # Competent report head: rebuild via domain logic from evidence.
            # Reward then reflects the *policy's* evidence coverage only when
            # coverage is complete; partial coverage yields partial submission
            # (simulates an agent that can only report what it inspected).
            report = self._build_evidence_report()
            return ToolAction(tool_name=name, arguments={"report": report})
        oid = order_id or (self._order_ids[0] if self._order_ids else "O1")
        if name in ("classify_order", "calculate_metrics", "get_order", "get_executions"):
            return ToolAction(tool_name=name, arguments={"order_id": oid})
        return ToolAction(tool_name=name, arguments={})

    def _build_evidence_report(self) -> str:
        """Report head: full gt builder if fully covered, else degraded pipe.

        Honest by construction: partial evidence -> partial report ->
        partial dense reward. Teaches the policy that coverage matters.
        """
        n = len(self._order_ids)
        full = n > 0 and all(
            o in self.classified and o in self.measured for o in self._order_ids
        )
        if full:
            return run_scenario_and_serialize(self.env.scenario, format="pipe")
        # Degraded: header only (+ whatever rows evidence justifies = none).
        # This yields dense reward ~0.05 (header) instead of 1.0.
        header = self.env.ground_truth_pipe.split("\n")[0]
        return header
