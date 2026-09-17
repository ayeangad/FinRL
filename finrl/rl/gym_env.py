"""Gymnasium-compatible Rule 605 tool-use environment.

MDP definition:
  state: scenario orders (fixed-size padded features) + step + per-slot
         evidence mask (which order slots have been classified / measured).
  action: MultiDiscrete [tool_idx, order_slot].
          tool_idx indexes into MASKED_TOOLS (default) or TOOLS (full).
          order_slot indexes into the scenario's order list (padded to
          MAX_ORDERS=32). submit_report ignores the slot.
          Legacy int/dict/ToolAction inputs remain accepted for backward
          compatibility with the pre-gym harness and tests.
  reward: shaping per step (+0.05 first classify/metrics per order,
          -0.01 redundant, -0.05 invalid/unknown order, -0.005 step cost,
          info tools neutral + step cost) + dense terminal reward from
          dense_report_reward on submit, or -0.1 truncation penalty.
  submit: evidence-conditioned composer. The report is built ONLY from
          OrderReports cached from tools the agent actually called during
          this episode (no re-read of the ground-truth scenario). Full
          coverage -> report equals ground truth; partial coverage ->
          partial report -> partial dense reward. This is honest by
          construction: no free ground-truth copy.

Registered as:
  gym.make("finrl/Rule605Tool-v0")            (masked, default)
  gym.make("finrl/Rule605ToolFull-v0")        (full 7-tool space)

Example:
  import gymnasium as gym
  import finrl.rl  # registers envs
  env = gym.make("finrl/Rule605Tool-v0", max_steps=20)
  obs, info = env.reset(seed=0, options={"scenario": "scenarios/v0.1/golden/market_01.json"})
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
    _GymBase = gym.Env
except Exception:  # pragma: no cover - gym-less fallback for docs/tests
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _GYM_AVAILABLE = False
    _GymBase = object  # type: ignore

from finrl.domain.order import OrderType
from finrl.env.rule_605_env import Rule605Env
from finrl.env.tools import ToolAction
from finrl.rl.reward import dense_report_reward
from finrl.rules.report_builder import build_rule_605_report
from finrl.rules.serializer import serialize_rule_605_pipe_delimited

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
TRUNCATION_PENALTY = -0.1

MAX_ORDERS = 32
ORDER_FEATURE_DIM = 8

# OrderType -> index for one-hot-ish encoding (4 known types).
_ORDER_TYPE_INDEX = {
    OrderType.MARKET: 0,
    OrderType.LIMIT: 1,
    OrderType.STOP: 2,
    OrderType.STOP_LIMIT: 3,
}


def _order_features(order, max_qty: float = 10000.0) -> list[float]:
    """Encode one order into ORDER_FEATURE_DIM floats in [0, 1]."""
    side = 1.0 if str(getattr(order, "side", "buy")).lower() == "buy" else 0.0
    otype = getattr(order, "order_type", None)
    t_idx = _ORDER_TYPE_INDEX.get(otype, -1)
    onehot = [0.0, 0.0, 0.0, 0.0]
    if 0 <= t_idx < 4:
        onehot[t_idx] = 1.0
    try:
        qty = float(getattr(order, "quantity", 0))
    except Exception:
        qty = 0.0
    log_qty = math.log10(max(1.0, qty)) / math.log10(max(2.0, max_qty))
    log_qty = float(min(1.0, max(0.0, log_qty)))
    has_limit = 1.0 if getattr(order, "limit_price", None) is not None else 0.0
    has_stop = 1.0 if getattr(order, "stop_price", None) is not None else 0.0
    return [side, *onehot, log_qty, has_limit, has_stop]


class Rule605GymEnv(_GymBase):  # type: ignore
    """Gymnasium tool-use wrapper. The underlying env stays source of truth."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 50,
        action_set: str = "masked",
        max_orders: int = MAX_ORDERS,
        submit_requires_evidence: bool = True,
    ):
        if _GYM_AVAILABLE:
            super().__init__()
        if action_set not in ("masked", "full"):
            raise ValueError("action_set must be 'masked' or 'full'")
        self.gym_max_steps = max_steps
        self.action_set = action_set
        self.action_names = MASKED_TOOLS if action_set == "masked" else TOOLS
        self.max_orders = int(max_orders)
        self.submit_requires_evidence = submit_requires_evidence
        self.env = Rule605Env(max_steps=max_steps)
        self.classified: set[str] = set()
        self.measured: set[str] = set()
        # Evidence cache: order_id -> OrderReport gathered via tools this episode.
        # Submit composes ONLY from this cache (never re-reads the scenario).
        self._evidence: dict[str, Any] = {}
        self._order_ids: list[str] = []
        self.action_space_n = len(self.action_names)
        self.np_random: Any = None
        self._seed: int | None = None
        # Gym spaces (fixed-size, SB3-compatible: Dict of Box).
        if _GYM_AVAILABLE:
            n_tools = len(self.action_names)
            self.action_space = spaces.MultiDiscrete([n_tools, self.max_orders])
            self.observation_space = spaces.Dict(
                {
                    "n_orders": spaces.Box(low=0, high=32, shape=(1,), dtype=np.float32),
                    "step": spaces.Box(low=0, high=10_000, shape=(1,), dtype=np.float32),
                    "coverage": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "order_features": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(self.max_orders, ORDER_FEATURE_DIM), dtype=np.float32,
                    ),
                    "evidence_mask": spaces.Box(
                        low=0.0, high=1.0, shape=(self.max_orders, 2), dtype=np.float32,
                    ),
                }
            )
        else:
            self.action_space = None
            self.observation_space = None

    # -- gym API --------------------------------------------------------
    def reset(self, scenario_or_path=None, seed: int | None = None, options: dict | None = None):
        if scenario_or_path is None and options and "scenario" in options:
            scenario_or_path = options["scenario"]
        if scenario_or_path is None:
            # Gymnasium protocol: reset() with no args must work (for
            # check_env / VecEnv). Fall back to a bundled default scenario.
            default = Path("scenarios/v0.1/golden/market_01.json")
            if default.exists():
                scenario_or_path = default
            else:
                goldens = sorted(Path("scenarios/v0.1/golden").glob("*.json")) if Path("scenarios/v0.1/golden").exists() else []
                if goldens:
                    scenario_or_path = goldens[0]
        if scenario_or_path is None:
            raise ValueError("reset() requires a scenario path or Scenario")
        # Seeding (gymnasium protocol).
        if seed is not None:
            self._seed = int(seed)
            if _GYM_AVAILABLE:
                try:
                    self.np_random, _ = gym.utils.seeding.np_random(int(seed))
                except Exception:
                    self.np_random = np.random.default_rng(int(seed))
            else:
                self.np_random = np.random.default_rng(int(seed))
        elif self.np_random is None:
            self.np_random = np.random.default_rng()
        obs = self.env.reset(scenario_or_path)
        self.classified = set()
        self.measured = set()
        self._evidence = {}
        self._order_ids = [o.order_id for o in self.env.scenario.get_all_orders()]
        out = self._obs_dict()
        info = {
            "scenario_id": obs.scenario_id,
            "n_orders": len(self._order_ids),
            "coverage": float(len(self.classified & self.measured) / max(1, len(self._order_ids))),
            "classified": [],
            "measured": [],
        }
        # Always return (obs, info) tuple per gymnasium API.
        return out, info

    def _obs_dict(self) -> dict[str, Any]:
        n = len(self._order_ids)
        covered = len(self.classified & self.measured)
        orders = self.env.scenario.get_all_orders() if self.env.scenario is not None else []
        feats = np.zeros((self.max_orders, ORDER_FEATURE_DIM), dtype=np.float32)
        for i, o in enumerate(orders[: self.max_orders]):
            feats[i] = np.asarray(_order_features(o), dtype=np.float32)
        mask = np.zeros((self.max_orders, 2), dtype=np.float32)
        id_to_slot = {oid: i for i, oid in enumerate(self._order_ids[: self.max_orders])}
        for oid in self.classified:
            if oid in id_to_slot:
                mask[id_to_slot[oid], 0] = 1.0
        for oid in self.measured:
            if oid in id_to_slot:
                mask[id_to_slot[oid], 1] = 1.0
        return {
            "n_orders": np.asarray([float(n)], dtype=np.float32),
            "step": np.asarray([float(self.env.current_step)], dtype=np.float32),
            "coverage": np.asarray([float(covered / max(1, n))], dtype=np.float32),
            "order_features": feats,
            "evidence_mask": mask,
        }

    def legacy_state(self) -> dict[str, Any]:
        """Backward-compat view: classified/measured lists + done flag."""
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
        """Fixed 5-dim feature vector for the linear policy (legacy)."""
        o = self._obs_dict()
        n = float(np.asarray(o["n_orders"]).ravel()[0])
        step = float(np.asarray(o["step"]).ravel()[0])
        cov = float(np.asarray(o["coverage"]).ravel()[0])
        n_norm = min(n, 32) / 32.0
        s_norm = min(step, self.gym_max_steps) / max(1, self.gym_max_steps)
        return [1.0, n_norm, s_norm, float(cov), float(cov * s_norm)]

    def step(self, action: int | dict | ToolAction | Any, order_id: str | None = None):
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
        # Cache evidence from privileged tools so submit composes from
        # what the agent actually observed (never from GT re-read).
        if tool_action.tool_name in ("classify_order", "calculate_metrics"):
            self._cache_evidence(tool_action)
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
            reward = TRUNCATION_PENALTY
            result.info["dense"] = {"reward": 0.0}
        else:
            reward = float(shaped)
            result.reward = reward

        obs = self._obs_dict()
        # Always gymnasium 5-tuple.
        return obs, reward, terminated, truncated, result.info

    # -- helpers --------------------------------------------------------
    def _slot_to_order_id(self, slot: int) -> str | None:
        if not self._order_ids:
            return None
        idx = int(slot) % max(1, len(self._order_ids))
        return self._order_ids[idx]

    def _coerce_action(self, action, order_id=None) -> ToolAction:
        # Direct ToolAction passthrough.
        if isinstance(action, ToolAction):
            return action
        # Dict-style tool call.
        if isinstance(action, dict):
            return ToolAction(
                tool_name=action.get("tool_name", "get_order"),
                arguments=action.get("arguments", {}),
            )
        # New MultiDiscrete style: [tool_idx, slot] array/tuple/list.
        if isinstance(action, (list, tuple, np.ndarray)) and not isinstance(action, str):
            try:
                arr = list(action)
            except Exception:
                arr = [action]
            if len(arr) >= 2:
                try:
                    t_idx = int(arr[0]) % len(self.action_names)
                    slot = int(arr[1])
                    name = self.action_names[t_idx]
                    if name == "submit_report":
                        report = self._build_evidence_report()
                        return ToolAction(tool_name=name, arguments={"report": report})
                    oid = self._slot_to_order_id(slot) or (
                        order_id or (self._order_ids[0] if self._order_ids else "O1")
                    )
                    if name in ("classify_order", "calculate_metrics", "get_order", "get_executions"):
                        return ToolAction(tool_name=name, arguments={"order_id": oid})
                    return ToolAction(tool_name=name, arguments={})
                except (ValueError, TypeError, IndexError):
                    pass
            if len(arr) == 1:
                return self._coerce_action(arr[0], order_id)
        # Legacy int: tool index + explicit order_id side-channel.
        try:
            idx = int(action)
        except (ValueError, TypeError):
            return ToolAction(tool_name="get_order", arguments={})
        if self.action_set == "masked":
            name = self.action_names[idx % len(self.action_names)]
            if name == "submit_report":
                report = self._build_evidence_report()
                return ToolAction(tool_name=name, arguments={"report": report})
            oid = order_id or (self._order_ids[0] if self._order_ids else "O1")
            return ToolAction(tool_name=name, arguments={"order_id": oid})
        name = TOOLS[idx % len(TOOLS)]
        if name == "submit_report":
            report = self._build_evidence_report()
            return ToolAction(tool_name=name, arguments={"report": report})
        oid = order_id or (self._order_ids[0] if self._order_ids else "O1")
        if name in ("classify_order", "calculate_metrics", "get_order", "get_executions"):
            return ToolAction(tool_name=name, arguments={"order_id": oid})
        return ToolAction(tool_name=name, arguments={})

    def _cache_evidence(self, tool_action: ToolAction) -> None:
        """Reconstruct and cache the OrderReport for the touched order.

        Uses the same domain logic as the underlying env's tool, but the
        cached object is what submit composes from. Information flows
        through the agent's tool calls, not through a submit-time
        re-read of the scenario.
        """
        try:
            from finrl.evals.order_evaluator import evaluate_order
            from finrl.scenario_runner import parse_scenario
        except Exception:
            return
        oid = tool_action.arguments.get("order_id")
        if not oid or self.env.scenario is None:
            return
        try:
            domain_orders, market, domain_execs = parse_scenario(self.env.scenario)
        except Exception:
            return
        target = next((o for o in domain_orders if o.order_id == oid), None)
        if target is None:
            return
        target_execs = [e for e in domain_execs if e.order_id == oid]
        try:
            report = evaluate_order(target, target_execs, market)
        except Exception:
            return
        # Only cache once both evidence types have been gathered, so the
        # composer reflects true coverage. Store the full report either way
        # but gate submit on the classified & measured sets.
        existing = self._evidence.get(oid)
        if existing is None:
            self._evidence[oid] = report
        else:
            self._evidence[oid] = report

    def _build_evidence_report(self) -> str:
        """Evidence-conditioned composer (no ground-truth re-read).

        Builds a Rule 605 pipe from cached OrderReports for orders where
        the agent gathered BOTH classify and measured evidence. Partial
        evidence -> partial report -> partial dense reward. No evidence ->
        header only (~0.05).
        """
        from finrl.rules.serializer import PIPE_DELIMITED_HEADER

        covered_ids = [o for o in self._order_ids if o in self.classified and o in self.measured]
        reports = [self._evidence[o] for o in covered_ids if o in self._evidence]
        if not reports:
            header = "|".join(PIPE_DELIMITED_HEADER)
            return header
        try:
            built = build_rule_605_report(reports)
            pipe = serialize_rule_605_pipe_delimited(built)
            # If no reportable rows among covered evidence, the builder still
            # returns a header-only-equivalent report; return as-is so the
            # dense reward reflects it honestly.
            return pipe
        except Exception:
            header = "|".join(PIPE_DELIMITED_HEADER)
            return header

    # -- gym housekeeping -------------------------------------------------
    def render(self):
        return None

    def close(self):
        return None
