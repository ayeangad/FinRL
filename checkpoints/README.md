# Checkpoints

Reproducible training outputs (no external weights). Regeneration commands
are listed per row; evaluation protocol and citations: `docs/gym_env.md`,
`docs/REFERENCES.md`.

| File | Produced by | Recorded outcome |
|------|-------------|------------------|
| `tool_policy.npz`, `tool_policy_curve.json` | `python -m finrl.rl.train --episodes 300` (numpy REINFORCE [3], CPU, <2 s) | Sampled-rollout train dense 1.000; test dense 0.584 (see `experiments/rl/eval_tool_test.json`). Greedy decoding collapses; the harness evaluates by sampling. |
| `ppo_tool.zip`, `ppo_tool_curve.json` | `python -m finrl.rl.train_sb3 --env tool --timesteps 50000 --seed 0` (PPO [4] via SB3 [5], CPU, ~140 s) | Test dense 1.000, success 1.000; val-gated with `EvalCallback`. |
| `ppo_compose.zip`, `ppo_compose_curve.json` | `python -m finrl.rl.train_sb3 --env compose --timesteps 50000 --seed 0` | Test dense 0.050, success 0.000. Retained as a negative result. |
| `rlvr_dataset.jsonl` (72 rows) | `export_rlvr_dataset(split='train')` (`finrl/rl/dataset.py`) | Train-split `{prompt, ground_truth_pipe}` pairs; report text excluded from prompts. Consumed by `finrl/rl/train_grpo_stub.py` with `dense_report_reward` as the outcome reward [7][8]. |

`experiments/rl/` holds the corresponding eval JSON. `checkpoints/sb3_eval/`
(intermediate SB3 logs) is gitignored.
