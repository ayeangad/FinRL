# Checkpoints — lineage

Owned training artifacts. Every file is reproducible from the repo (no external weights).

| File | How produced | What it proves |
|------|--------------|----------------|
| `tool_policy.npz` + `tool_policy_curve.json` | `python -m finrl.rl.train --episodes 300` (numpy REINFORCE, CPU ~1s) | Stochastic tool-ordering baseline learns; sampled rollouts reach `mean_dense=1.0` train |
| `ppo_tool.zip` + `ppo_tool_curve.json` | `python -m finrl.rl.train_sb3 --env tool --timesteps 50000 --seed 0` (SB3 PPO, CPU ~140s) | Classic RL solves tool-use: `test mean_dense=0.9986, success=1.000`, train→test gap `0.0005` |
| `ppo_compose.zip` + `ppo_compose_curve.json` | `python -m finrl.rl.train_sb3 --env compose --timesteps 50000 --seed 0` | Composition is hard for tabular PPO: `test mean_dense=0.05` — motivates the LLM/GRPO track |
| `rlvr_dataset.jsonl` (72 items, train split) | `python -c "from finrl.rl.dataset import export_rlvr_dataset; export_rlvr_dataset(split='train')"` | RLVR seam: `{prompt, ground_truth_pipe}` with GT never in prompt; reward = `dense_report_reward` |

Eval artifacts: `experiments/rl/eval_tool_{train,val,test}.json`, `eval_compose_test.json`, `splits.json`.
SB3 intermediate evals (`checkpoints/sb3_eval/`) are gitignored.
