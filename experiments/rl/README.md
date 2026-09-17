# RL experiments — what to read first

- `splits.json` — hash-stable 72/12/16 train/val/test over `scenarios/v0.1/golden/`.
- `eval_tool_train.json`, `eval_tool_val.json`, `eval_tool_test.json` — random / REINFORCE / PPO on the Tool MDP.
- `eval_compose_test.json` — random / PPO on the Compose MDP (hard; documents the LLM-track motivation).

Headline (test split, Tool MDP, `max_steps=12`):

| policy | mean_dense | success | steps |
|--------|------------|---------|-------|
| random | 0.406 | 0.375 | 3.0 |
| reinforce (linear, sampled) | 0.584 | 0.562 | 2.2 |
| ppo (`ppo_tool.zip`) | 0.999 | 1.000 | 3.5 |

Compose MDP (test): random `0.003`, PPO `0.050` — 30-way category×bucket prediction from scratch is not solved by 50k steps of tabular PPO; use the RLVR/GRPO track (`python -m finrl.rl.train_grpo_stub --smoke`) for composition.

Reproduce:
```bash
./.venv/bin/python -m finrl.rl.evaluate --env tool --split test --policies random,reinforce,ppo --ppo-path checkpoints/ppo_tool.zip
./.venv/bin/python -m finrl.rl.evaluate --env compose --split test --policies random,ppo --ppo-path checkpoints/ppo_compose.zip
```
