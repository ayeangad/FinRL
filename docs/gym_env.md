# Rule 605 Gym Environments — Env Card

Two registered Gymnasium MDPs over SEC Rule 605 reporting scenarios. Both are
deterministic simulators with fixed-size `Dict` observations and `MultiDiscrete`
actions, SB3-compatible (`MultiInputPolicy`).

```python
import finrl.rl  # registers envs
import gymnasium as gym
env = gym.make("finrl/Rule605Tool-v0", max_steps=12)
obs, info = env.reset(seed=0, options={"scenario": "scenarios/v0.1/golden/market_01.json"})
```

## Env 1 — `finrl/Rule605Tool-v0` (masked) / `finrl/Rule605ToolFull-v0` (full)

**Task:** gather per-order evidence efficiently, then submit.

| | |
|---|---|
| Action | `MultiDiscrete([3, 32])` masked (`[7, 32]` full). `[tool_idx, order_slot]`. Masked tools: `classify_order, calculate_metrics, submit_report`. Full adds `get_order, get_quote, get_quotes, get_executions`. Legacy `step(int, order_id=)` still accepted. |
| Observation | `Dict(n_orders: Box(1), step: Box(1), coverage: Box(1), order_features: Box(32,8), evidence_mask: Box(32,2))`. `order_features` = `[side, type_onehot(4), log_qty, has_limit, has_stop]`. |
| Reward | `+0.05` first classify/metrics per order, `-0.01` redundant, `-0.05` invalid, `-0.005` step cost, `-0.1` truncation. Submit → `dense_report_reward` in `[0,1]`. |
| Submit | **Evidence-conditioned composer:** builds the pipe ONLY from `OrderReport`s cached from tools called this episode. Full coverage → equals ground truth; partial → partial dense reward; none → header-only (~0.05). Never re-reads the scenario. |
| Termination | `terminated` on submit, `truncated` at `max_steps`. Always returns `(obs, reward, terminated, truncated, info)` with `info["dense"]` on submit. |
| Seeding | `reset(seed, options={"scenario": path})`. No-arg `reset()` falls back to `market_01.json` (for `check_env` / VecEnv). |

## Env 2 — `finrl/Rule605Compose-v0`

**Task:** compose the report's categorical structure yourself (no oracle tools).

| | |
|---|---|
| Action | `MultiDiscrete([2, 32, 30])` = `[phase, slot, value]`. Phase 0: info tool `value 0..4 → [get_order, get_quote, get_quotes, get_executions, submit_report]`. Phase 1: predict `value 0..29 → category(5) × bucket(6)` combo. Privileged `classify/calculate` tools are absent by design. |
| Observation | Tool obs + `grounded_mask(32,1)` (touched via get_order/get_executions), `pred_mask(32,1)`, `pred_combo(32,1)` (own staged predictions, normalized). Quotes are NOT observed — query them. |
| Reward | Predict `+0.05` first-correct, `-0.02` wrong, `-0.01` re-predict, `-0.05` invalid slot, `-0.005` step cost. Submit → dense reward on pipe assembled from predictions + true numeric metrics, **gated on grounding** (predicted-but-untouched rows excluded). Reward may read GT; observations never do. |
| Why | Classification (`marketable` vs `midpoint` vs `stop`) needs quote context, so the agent must ground (touch orders/quotes) AND learn the mapping. |

## Splits & generalization

Hash-stable 70/15/15 over `scenarios/v0.1/golden/*.json` (72/12/16): see `finrl/rl/splits.py`, artifact `experiments/rl/splits.json`. Train on `train`, tune on `val`, report on `test`. Synthetic augmentation: `python -m finrl.scenario_gen --n 200 --seed 0 --out-dir scenarios/v0.1/synth`.

## Training tracks

- **Classic:** `python -m finrl.rl.train_sb3 --env tool --timesteps 50000` (PPO, `MultiInputPolicy`, train-split randomization, val-gated). Legacy REINFORCE: `python -m finrl.rl.train --episodes 200`.
- **RLVR/LLM:** `python -m finrl.rl.train_grpo_stub --smoke` (CPU) or `--train` with TRL. Dataset: `checkpoints/rlvr_dataset.jsonl` (train-split, GT never in prompt). Reward fn: `finrl.rl.reward.dense_report_reward`.
- **Eval:** `python -m finrl.rl.evaluate --env tool --split test --policies random,reinforce,ppo`.

## Files

- `finrl/rl/gym_env.py` — Tool MDP + evidence-conditioned composer
- `finrl/rl/compose_env.py` — Composition MDP
- `finrl/rl/splits.py`, `finrl/scenario_gen.py` — generalization
- `finrl/rl/train_sb3.py`, `finrl/rl/evaluate.py`, `finrl/rl/train_grpo_stub.py`, `finrl/rl/demo.py`
