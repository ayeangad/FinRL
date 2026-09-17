# RL experiments — what to read first

- `splits.json` — hash-stable 72/12/16 train/val/test over `scenarios/v0.1/golden/`.
- `eval_tool_train.json`, `eval_tool_val.json`, `eval_tool_test.json` — random / REINFORCE / PPO on the Tool MDP.
- `eval_compose_test.json` — random / PPO on the Compose MDP (hard; documents the LLM-track motivation).

Headline (test split, Tool MDP, `max_steps=12`, deterministic `seed=0`):

| policy | mean_dense | success | steps |
|--------|------------|---------|-------|
| random | 0.286 | 0.250 | 2.2 |
| reinforce (linear, sampled) | 0.584 | 0.562 | 2.2 |
| ppo (`ppo_tool.zip`) | 1.000 | 1.000 | 3.3 |

Train→test gap (PPO): train `0.999` → test `1.000` (no overfit; val `1.000`).

Compose MDP (test): random `0.006`, PPO `0.050` — 30-way category×bucket prediction from scratch is not solved by 50k steps of tabular PPO; use the RLVR/GRPO track (`python -m finrl.rl.train_grpo_stub --smoke`) for composition.

Reproduce:
```bash
./.venv/bin/python -m finrl.rl.evaluate --env tool --split test --policies random,reinforce,ppo --ppo-path checkpoints/ppo_tool.zip
./.venv/bin/python -m finrl.rl.evaluate --env compose --split test --policies random,ppo --ppo-path checkpoints/ppo_compose.zip
```
