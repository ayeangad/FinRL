# Rule 605 environments — specification

Two episodic, finite-horizon MDPs [1] over SEC Rule 605 reporting scenarios [9],
implemented against the Gymnasium API [2]. Both are deterministic simulators;
stochasticity enters only through the initial scenario draw and the learning
algorithm. Full citation list: `docs/REFERENCES.md`.

```python
import finrl.rl  # registers finrl/* ids
import gymnasium as gym
env = gym.make("finrl/Rule605Tool-v0", max_steps=12)
obs, info = env.reset(seed=0, options={"scenario": "scenarios/v0.1/golden/market_01.json"})
```

## 1. Common setup

Scenarios are static JSON records (orders, NBBO quotes, executions) plus a
ground-truth pipe report derived from the domain rules in `finrl/rules/`.
An episode fixes one scenario, runs at most `max_steps`, and ends on
`submit_report` (`terminated=True`) or step budget exhaustion
(`truncated=True`). All environments return the Gymnasium 5-tuple
`(obs, reward, terminated, truncated, info)` and pass
`gymnasium.utils.env_checker.check_env` (`tests/test_gym_api.py`,
`tests/test_compose_env.py`).

Observations are fixed-size `Dict` spaces of `Box` arrays (SB3-compatible
with `MultiInputPolicy` [5]). Order features per slot are
`[side, type_onehot(4), log_qty, has_limit, has_stop]`, all in [0, 1],
zero-padded to 32 slots. Rewards are in two parts: small per-step shaping
terms and a terminal outcome reward in [0, 1] from `dense_report_reward`
(per-column partial credit with numeric tolerance; `finrl/rl/reward.py`).

## 2. `finrl/Rule605Tool-v0` (masked) / `finrl/Rule605ToolFull-v0` (full)

Tests evidence-gathering efficiency: which orders to inspect, in what order,
and when to submit.

- Action: `MultiDiscrete([3, 32])` masked, `[7, 32]` full. Element 0 is the
  tool index, element 1 the order slot (modulo `n_orders`). Masked tools are
  `classify_order, calculate_metrics, submit_report`; full adds `get_order,
  get_quote, get_quotes, get_executions`. Integer/dict/`ToolAction` inputs
  remain accepted for backward compatibility with the pre-Gym harness.
- Observation: `n_orders(1), step(1), coverage(1), order_features(32,8),
  evidence_mask(32,2)` (classified / measured per slot).
- Shaping: +0.05 first classify/metrics per order, −0.01 redundant, −0.05
  invalid order, −0.005 step cost, −0.1 truncation penalty.
- Submit: evidence-conditioned. The pipe is assembled exclusively from
  `OrderReport` objects cached from tools invoked in the current episode;
  orders lacking both classify and metrics evidence are excluded. Full
  coverage therefore reproduces ground truth; partial coverage yields
  partial credit; no coverage yields header only (~0.05). The submit path
  does not re-read the scenario (`tests/test_no_oracle.py`).

## 3. `finrl/Rule605Compose-v0`

Tests categorical composition under grounding. Privileged
`classify/calculate` tools are excluded by design, so copying answers is
not possible.

- Action: `MultiDiscrete([2, 32, 30])` = `[phase, slot, value]`. Phase 0
  selects an information tool (`0..4`: `get_order, get_quote, get_quotes,
  get_executions, submit_report`); phase 1 stages a prediction
  (`0..29`: `category(5) × bucket(6)`).
- Observation: tool observation plus `grounded_mask(32,1)` (slot touched via
  `get_order`/`get_executions`), `pred_mask(32,1)`, `pred_combo(32,1)`
  (staged prediction, normalized). NBBO quotes are not observed and must be
  queried.
- Shaping: +0.05 first correct prediction, −0.02 incorrect, −0.01
  re-prediction, −0.05 invalid slot, −0.005 step cost.
- Submit: rows are included only for predicted *and* grounded orders;
  numeric fields are filled from domain evaluation while the predicted
  category/bucket determine row placement. The reward function may read
  ground truth; observations never expose it.

## 4. Splits and evaluation protocol

Hash-stable 72/12/16 train/val/test split over `scenarios/v0.1/golden/`
(`finrl/rl/splits.py`; artifact `experiments/rl/splits.json`). Train on
train, select on val, report on test. `finrl/rl/evaluate.py` reports mean
and standard deviation of the dense outcome reward, success rate
(dense ≥ 0.95), and mean episode length, with per-scenario rows in the JSON
artifacts. Evaluation is seeded and deterministic
(`tests/test_eval_determinism.py`). Synthetic augmentation is available via
`finrl/scenario_gen.py` (seeded; every generated order is executable by
construction).

## 5. Training

- Tabular baseline: batched REINFORCE with mean-return baseline [3]
  (`finrl/rl/train.py`; linear softmax over 5-dim features). Included as a
  lower bound; greedy decoding from this policy is known to collapse and the
  harness evaluates it by sampling to match the training distribution.
- Primary: PPO [4] via Stable-Baselines3 [5]
  (`finrl/rl/train_sb3.py`; `MultiInputPolicy`, `DummyVecEnv`,
  `EvalCallback` on val). Observed: Tool-v0 reaches test dense 1.000
  (success 1.000) at 50k timesteps with train→test gap ≈ 0
  (`experiments/rl/`); Compose-v0 remains near floor at the same budget
  (test dense 0.050), consistent with a 30-way per-slot prediction problem
  under sparse outcome reward.
- Language-model track: RLVR [8] with GRPO-style [7] outcome rewards.
  `finrl/rl/dataset.py` exports train-split `{prompt, ground_truth_pipe}`
  pairs with the report excluded from the prompt;
  `finrl/rl/train_grpo_stub.py` provides a CPU smoke test of the reward seam
  and a TRL entry point when available.

## 6. Limitations

1. Shaping bonuses (§2-§3) are not potential-based in the sense of [6] and
   therefore carry no policy-invariance guarantee; terminal behavior is
   governed by the dense outcome reward, but shaping coefficients were tuned
   by hand and not ablated.
2. The 24-column report is a reduced subset of the Rule 605 NMS Plan
   specification [9] (currently 55 fields); stop/limit executability follows
   the repo's `finrl/rules/executable_time.py`, not a full regulatory
   implementation.
3. Scenarios are single-security, single-month, fully observed simulators.
   There is no market impact, partial observability beyond the evidence
   masks, or multi-agent interaction.
4. Compose-v0 fills numeric metrics from domain evaluation at submit time;
   the learned component is categorical placement, not spread arithmetic.
