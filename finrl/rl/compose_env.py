"""Gymnasium composition MDP: agent must predict category x bucket.

Why this env exists: Rule605GymEnv (tool-use) teaches *efficient evidence
gathering* but its submit composer fills numeric metrics from cached tool
outputs. This env teaches the harder skill: *composing the report's
categorical structure yourself*.

MDP:
  obs: order_features (32x8, always visible) + grounded_mask (did the agent
       touch this slot via get_order/get_executions?) + pred_mask/pred_combo
       (agent's own staged predictions). Quotes are NOT in the observation;
       the agent must query them via tools if it wants quote-aware
       classification (marketable vs midpoint logic).
  action: MultiDiscrete [phase, slot, value]
       phase 0 = info tool, value 0..4 ->
                 [get_order, get_quote, get_quotes, get_executions, submit_report]
       phase 1 = predict, value 0..29 -> category*6+bucket combo.
       Privileged classify/calculate tools are INTENTIONALLY absent: the
       agent cannot copy answers, it must learn the mapping.
  reward: predict +0.05 first-correct, -0.02 wrong, -0.01 re-predict,
          -0.05 invalid slot, -0.005 step cost. Submit -> dense_report_reward
          on a report assembled from predictions + true numeric metrics,
          gated on grounding (predicted but never touched -> excluded).
          Reward may read ground truth (standard); observations never do.

Registered as gym.make("finrl/Rule605Compose-v0").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
    _GymBase = gym.Env
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _GYM_AVAILABLE = False
    _GymBase = object  # type: ignore

from finrl.env.rule_605_env import Rule605Env
from finrl.env.tools import ToolAction
from finrl.rl.gym_env import MAX_ORDERS, ORDER_FEATURE_DIM, _order_features
from finrl.rl.reward import dense_report_reward
from finrl.rules.classification import OrderTypeCategory
from finrl.rules.order_size import OrderSizeBucket
from finrl.rules.report_builder import build_rule_605_report
from finrl.rules.serializer import serialize_rule_605_pipe_delimited

COMPOSE_TOOLS = ["get_order", "get_quote", "get_quotes", "get_executions", "submit_report"]
SUBMIT_TOOL_IDX = COMPOSE_TOOLS.index("submit_report")

CATEGORIES = list(OrderTypeCategory)
BUCKETS = list(OrderSizeBucket)
N_COMBOS = len(CATEGORIES) * len(BUCKETS)  # 5 * 6 = 30

PREDICT_BONUS = 0.05
WRONG_PENALTY = -0.02
REPREDICT_PENALTY = -0.01
INVALID_PENALTY = -0.05
STEP_COST = -0.005
TRUNCATION_PENALTY = -0.1


def combo_to_cat_bucket(combo: int) -> tuple[OrderTypeCategory, OrderSizeBucket]:
    combo = int(combo) % N_COMBOS
    return CATEGORIES[combo // len(BUCKETS)], BUCKETS[combo % len(BUCKETS)]


def cat_bucket_to_combo(cat: OrderTypeCategory, bucket: OrderSizeBucket) -> int:
    return CATEGORIES.index(cat) * len(BUCKETS) + BUCKETS.index(bucket)


class Rule605ComposeEnv(_GymBase):  # type: ignore
    """Composition MDP. See module docstring."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 50, max_orders: int = MAX_ORDERS):
        if _GYM_AVAILABLE:
            super().__init__()
        self.gym_max_steps = max_steps
        self.max_orders = int(max_orders)
        self.env = Rule605Env(max_steps=max_steps)
        self._order_ids: list[str] = []
        self.grounded: set[str] = set()
        self.predictions: dict[str, int] = {}
        self.np_random: Any = None
        # Ground-truth (category, bucket) per order, computed at reset for
        # REWARD ONLY (never shown in observations).
        self._gt_combo: dict[str, int] = {}
        if _GYM_AVAILABLE:
            self.action_space = spaces.MultiDiscrete([2, self.max_orders, 30])
            self.observation_space = spaces.Dict(
                {
                    "n_orders": spaces.Box(low=0, high=32, shape=(1,), dtype=np.float32),
                    "step": spaces.Box(low=0, high=10_000, shape=(1,), dtype=np.float32),
                    "coverage": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "order_features": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(self.max_orders, ORDER_FEATURE_DIM), dtype=np.float32,
                    ),
                    "grounded_mask": spaces.Box(
                        low=0.0, high=1.0, shape=(self.max_orders, 1), dtype=np.float32,
                    ),
                    "pred_mask": spaces.Box(
                        low=0.0, high=1.0, shape=(self.max_orders, 1), dtype=np.float32,
                    ),
                    "pred_combo": spaces.Box(
                        low=0.0, high=1.0, shape=(self.max_orders, 1), dtype=np.float32,
                    ),
                }
            )
        else:
            self.action_space = None
            self.observation_space = None

    # -- gym API ------------------------------------------------------
    def reset(self, scenario_or_path=None, seed: int | None = None, options: dict | None = None):
        if scenario_or_path is None and options and "scenario" in options:
            scenario_or_path = options["scenario"]
        if scenario_or_path is None:
            default = Path("scenarios/v0.1/golden/market_01.json")
            if default.exists():
                scenario_or_path = default
            else:
                goldens = sorted(Path("scenarios/v0.1/golden").glob("*.json")) if Path("scenarios/v0.1/golden").exists() else []
                if goldens:
                    scenario_or_path = goldens[0]
        if scenario_or_path is None:
            raise ValueError("reset() requires a scenario path or Scenario")
        if seed is not None:
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
        self._order_ids = [o.order_id for o in self.env.scenario.get_all_orders()]
        self.grounded = set()
        self.predictions = {}
        self._gt_combo = self._compute_gt_combos()
        out = self._obs_dict()
        info = {"scenario_id": obs.scenario_id, "n_orders": len(self._order_ids)}
        return out, info

    def _compute_gt_combos(self) -> dict[str, int]:
        try:
            from finrl.evals.order_evaluator import evaluate_order
            from finrl.scenario_runner import parse_scenario
        except Exception:
            return {}
        try:
            domain_orders, market, domain_execs = parse_scenario(self.env.scenario)
        except Exception:
            return {}
        out: dict[str, int] = {}
        for o in domain_orders:
            try:
                rep = evaluate_order(o, [e for e in domain_execs if e.order_id == o.order_id], market)
                out[o.order_id] = cat_bucket_to_combo(rep.order_type_category, rep.order_size_bucket)
            except Exception:
                continue
        return out

    def _obs_dict(self) -> dict[str, Any]:
        n = len(self._order_ids)
        orders = self.env.scenario.get_all_orders() if self.env.scenario is not None else []
        feats = np.zeros((self.max_orders, ORDER_FEATURE_DIM), dtype=np.float32)
        for i, o in enumerate(orders[: self.max_orders]):
            feats[i] = np.asarray(_order_features(o), dtype=np.float32)
        grounded_m = np.zeros((self.max_orders, 1), dtype=np.float32)
        pred_m = np.zeros((self.max_orders, 1), dtype=np.float32)
        pred_c = np.zeros((self.max_orders, 1), dtype=np.float32)
        id_to_slot = {oid: i for i, oid in enumerate(self._order_ids[: self.max_orders])}
        for oid in self.grounded:
            if oid in id_to_slot:
                grounded_m[id_to_slot[oid], 0] = 1.0
        for oid, combo in self.predictions.items():
            if oid in id_to_slot:
                pred_m[id_to_slot[oid], 0] = 1.0
                pred_c[id_to_slot[oid], 0] = float(combo) / max(1, N_COMBOS - 1)
        cov = len(self.predictions) / max(1, n)
        return {
            "n_orders": np.asarray([float(n)], dtype=np.float32),
            "step": np.asarray([float(self.env.current_step)], dtype=np.float32),
            "coverage": np.asarray([float(cov)], dtype=np.float32),
            "order_features": feats,
            "grounded_mask": grounded_m,
            "pred_mask": pred_m,
            "pred_combo": pred_c,
        }

    def _slot_to_order_id(self, slot: int) -> str | None:
        if not self._order_ids:
            return None
        if int(slot) < 0 or int(slot) >= len(self._order_ids):
            return None
        return self._order_ids[int(slot)]

    def step(self, action):
        try:
            arr = list(action)
        except Exception:
            arr = [0, 0, 0]
        phase = int(arr[0]) if len(arr) > 0 else 0
        slot = int(arr[1]) if len(arr) > 1 else 0
        value = int(arr[2]) if len(arr) > 2 else 0

        if phase == 1:
            return self._step_predict(slot, value)
        return self._step_tool(slot, value)

    def _step_tool(self, slot: int, value: int):
        if value < 0 or value >= len(COMPOSE_TOOLS):
            # Invalid tool index.
            self.env.current_step += 1
            reward = INVALID_PENALTY + STEP_COST
            if self.env.current_step >= self.env.max_steps and not self.env.done:
                self.env.done = True
                return self._obs_dict(), TRUNCATION_PENALTY, False, True, {"truncated": True}
            return self._obs_dict(), float(reward), False, False, {}
        name = COMPOSE_TOOLS[value]
        if name == "submit_report":
            report = self._build_predicted_report()
            res = self.env.step(ToolAction(tool_name=name, arguments={"report": report}))
            dense = dense_report_reward(report, self.env.ground_truth_pipe)
            res.info["dense"] = dense
            res.info["predictions"] = dict(self.predictions)
            terminated = res.done and "truncated" not in res.info
            truncated = bool(res.info.get("truncated", False))
            return self._obs_dict(), float(dense["reward"]), terminated, truncated, res.info
        oid = self._slot_to_order_id(slot)
        if oid is None:
            self.env.current_step += 1
            reward = INVALID_PENALTY + STEP_COST
            if self.env.current_step >= self.env.max_steps and not self.env.done:
                self.env.done = True
                return self._obs_dict(), TRUNCATION_PENALTY, False, True, {"truncated": True}
            return self._obs_dict(), float(reward), False, False, {}
        # Grounding tools.
        if name in ("get_order", "get_executions"):
            self.grounded.add(oid)
            res = self.env.step(ToolAction(tool_name=name, arguments={"order_id": oid}))
        elif name in ("get_quote", "get_quotes"):
            res = self.env.step(ToolAction(tool_name=name, arguments={}))
        else:
            res = self.env.step(ToolAction(tool_name=name, arguments={}))
        terminated = res.done and "truncated" not in res.info
        truncated = bool(res.info.get("truncated", False))
        if truncated:
            return self._obs_dict(), TRUNCATION_PENALTY, False, True, res.info
        return self._obs_dict(), float(STEP_COST), terminated, truncated, res.info

    def _step_predict(self, slot: int, combo: int):
        oid = self._slot_to_order_id(slot)
        if oid is None or combo < 0 or combo >= N_COMBOS:
            self.env.current_step += 1
            if self.env.current_step >= self.env.max_steps and not self.env.done:
                self.env.done = True
                return self._obs_dict(), TRUNCATION_PENALTY, False, True, {"truncated": True}
            return self._obs_dict(), float(INVALID_PENALTY + STEP_COST), False, False, {}
        self.env.current_step += 1
        first = oid not in self.predictions
        self.predictions[oid] = int(combo)
        gt = self._gt_combo.get(oid)
        if gt is None:
            reward = STEP_COST
        elif int(combo) == int(gt):
            reward = (PREDICT_BONUS if first else 0.0) + STEP_COST
            if not first:
                reward = STEP_COST  # re-predicting correctly is neutral
        else:
            reward = (WRONG_PENALTY if first else REPREDICT_PENALTY) + STEP_COST
        if self.env.current_step >= self.env.max_steps and not self.env.done:
            self.env.done = True
            return self._obs_dict(), float(reward) if reward != TRUNCATION_PENALTY else TRUNCATION_PENALTY, False, True, {"truncated": True}
        info = {"predicted": oid, "combo": int(combo), "correct": (gt is not None and int(combo) == int(gt))}
        return self._obs_dict(), float(reward), False, False, info

    def _build_predicted_report(self) -> str:
        """Assemble pipe from predictions + true numeric metrics (reward-side).

        Only orders that are BOTH predicted AND grounded are included, so an
        agent cannot score without touching the scenario. Numeric metrics are
        filled from domain evaluation (the categorical structure is what is
        learned here); ungrounded/ unpredicted orders are omitted, yielding
        honest partial credit via dense_report_reward.
        """
        from finrl.evals.order_evaluator import evaluate_order
        from finrl.rules.report import OrderReport
        from finrl.scenario_runner import parse_scenario

        try:
            domain_orders, market, domain_execs = parse_scenario(self.env.scenario)
        except Exception:
            from finrl.rules.serializer import PIPE_DELIMITED_HEADER

            return "|".join(PIPE_DELIMITED_HEADER)
        by_id = {o.order_id: o for o in domain_orders}
        execs_by_id: dict[str, list] = {}
        for e in domain_execs:
            execs_by_id.setdefault(e.order_id, []).append(e)
        staged: list[OrderReport] = []
        for oid, combo in self.predictions.items():
            if oid not in self.grounded:
                continue
            target = by_id.get(oid)
            if target is None:
                continue
            try:
                true_rep = evaluate_order(target, execs_by_id.get(oid, []), market)
            except Exception:
                continue
            cat, bucket = combo_to_cat_bucket(combo)
            staged.append(
                OrderReport(
                    order_id=true_rep.order_id,
                    order_size_bucket=bucket,
                    order_type_category=cat,
                    reportable=true_rep.reportable,
                    requested_quantity=true_rep.requested_quantity,
                    executed_quantity=true_rep.executed_quantity,
                    average_execution_price=true_rep.average_execution_price,
                    price_improvement=true_rep.price_improvement,
                    effective_spread=true_rep.effective_spread,
                    quoted_spread=true_rep.quoted_spread,
                    realized_spreads=true_rep.realized_spreads,
                    percentage_effective_spread=true_rep.percentage_effective_spread,
                    percentage_quoted_spread=true_rep.percentage_quoted_spread,
                    percentage_realized_spreads=true_rep.percentage_realized_spreads,
                    shares_price_improved=true_rep.shares_price_improved,
                    shares_at_quote=true_rep.shares_at_quote,
                    shares_outside_quote=true_rep.shares_outside_quote,
                )
            )
        if not staged:
            from finrl.rules.serializer import PIPE_DELIMITED_HEADER

            return "|".join(PIPE_DELIMITED_HEADER)
        try:
            built = build_rule_605_report(staged)
            return serialize_rule_605_pipe_delimited(built)
        except Exception:
            from finrl.rules.serializer import PIPE_DELIMITED_HEADER

            return "|".join(PIPE_DELIMITED_HEADER)

    def render(self):
        return None

    def close(self):
        return None
