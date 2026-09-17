# RL experiments

Protocol: hash-stable 72/12/16 train/val/test split (`splits.json`);
seeded evaluation (`finrl/rl/evaluate.py`); metrics are mean (std) dense
outcome reward, success rate at dense ≥ 0.95, and mean episode length.
Environment definitions and known limitations: `docs/gym_env.md`.
Citations: `docs/REFERENCES.md`.

Artifacts:

- `splits.json` — split membership.
- `eval_tool_train.json`, `eval_tool_val.json`, `eval_tool_test.json` — Tool MDP.
- `eval_compose_test.json` — Compose MDP.

Observed outcomes (Tool MDP, `max_steps=12`, `seed=0`):

| split | random | REINFORCE [3] | PPO [4] |
|-------|--------|---------------|---------|
| train (n=72) | 0.353 (0.443) | — | 0.999 (0.005) |
| val (n=12) | 0.288 (0.411) | 0.604 (0.468) | 1.000 (0.000) |
| test (n=16) | 0.286 (0.409) | 0.584 (0.471) | 1.000 (0.000) |

Success rates (test): random 0.250, REINFORCE 0.562, PPO 1.000. The
train→test difference for PPO is within measurement noise on this split.

Compose MDP (test, n=16): random 0.006 (0.017), PPO 0.050 (0.000),
success 0.000 for both. Per-scenario rows are in `eval_compose_test.json`.
This is consistent with a 30-way per-slot prediction task under sparse
outcome reward at 50k timesteps; the language-model (RLVR [8]) path is the
documented next step, not a claim made here.

Reproduce:

```bash
./.venv/bin/python -m finrl.rl.evaluate --env tool --split test --policies random,reinforce,ppo --ppo-path checkpoints/ppo_tool.zip
./.venv/bin/python -m finrl.rl.evaluate --env compose --split test --policies random,ppo --ppo-path checkpoints/ppo_compose.zip
```
