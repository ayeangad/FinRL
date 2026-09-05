# Phase 10: Context Orchestration

Goal: stop re-sending unbounded conversation history and monolithic rule text at
every step. Instead: bound history with a token-budget window and select only the
rule sections relevant to each scenario. This is a retrieval/context-selection
layer in front of the model.

## Changes

- `finrl/benchmark/agent.py`
  - `ConversationWindow`: bounded, recency-aware history. Keeps the last N
    (model, tool) exchange pairs unconditionally and fills the rest of the token
    budget oldest-first; older entries are pruned and counted.
  - Both `QwenAgent` and `OpenAIAgent` use `ConversationWindow` instead of an
    unbounded `conversation_history` list (default budgets: 3000 / 6000 tokens).
  - Both agents build their system prompt through `ContextSelector` by default
    (`use_scenario_context=True`); fall back to the monolithic prompt if the
    sections directory is absent. `prompt_strategy` and per-run context metrics
    are recorded in each trace.
- `finrl/benchmark/context_selector.py` (new)
  - `ScenarioProfile` extracts, from each scenario's orders, the order types,
    count, and whether it spans multiple orders.
  - `ContextSelector` always includes task / tools / schema / formatting / ReAct
    sections, then adds only the rule sections for the order types present
    (stop, market, limit) plus multi-order aggregation rules when relevant.
- `prompts/sections/*.txt` (new) — modular rule sections (00-08).
- `finrl/env/rule_605_env.py` — `action_history` capped (default 20 actions) so
  observations/traces do not grow unboundedly.
- `finrl/benchmark/__main__.py` — `--prompt-strategy {scenario_aware,full}`.
- tests: `tests/test_conversation_window.py`, `tests/test_context_selector.py`.

## Offline replay of phase-9 real traces (before vs after)

Replayed the 302 pre-existing real traces through `ConversationWindow` and the
selector (token estimate ~1 token / 4 chars):

```
All traces (302):          peak unbounded avg=1319t | bounded avg=1319t
Multi-step traces (99):    peak unbounded avg=2123t | bounded avg=2123t
```

Budget sensitivity (multi-step >= 4 steps, avg bounded peak):

```
budget | avg full | avg bounded | pct of full | avg dropped
   500 |     2123 |        1879 |       88.5% |        2.1
  1000 |     2123 |        1896 |       89.3% |        1.7
  1500 |     2123 |        2106 |       99.2% |        0.1
  2000 |     2123 |        2123 |      100.0% |        0.0
  3000 |     2123 |        2123 |      100.0% |        0.0
```

Reference points from the phase-9 `openai_real` traces (model-reported input
tokens, cumulative over the run): `edge_case_10` = 14,198 by step 9,
`edge_case_01` = 9,676, `edge_case_02` = 9,793. Under the phase-10 window the
per-step prompt is capped at the budget, so those numbers stop growing.

## Scenario-aware prompt sizes

The monolithic prompt (`rule_605_v1.txt`) is ~768 tokens. Selected sections:

```
market_01.json / market_05.json / multi_exec_01 / edge_case_07  -> 6 sections, 835 tokens
stop_01.json / stop_limit_01                                    -> 6 sections, 866 tokens
marketable / midpoint / non_marketable limit                    -> 6 sections, 876 tokens
```

The base sections embed the same always-needed instruction text as the original
prompt; the conditional sections add the order-type specific rules. Savings grow
as the rulebook/section library grows — this is the retrieval seam.

## Real Qwen3-0.6B "after" run

Controlled 5-scenario run (`real_5scenario.txt`; `max_history_tokens=3000`,
per-scenario subprocess, 600 s wall-cap):

```
scenario                 type    termination steps input_tok peak_prompt  trunc  dropped elapsed_s  score
market_01.json           market  submitted      4      6615      1888      0        0     54.9  0.000
marketable_limit_01.json limit   timeout       -1         0         0      0        0      600  0.000
midpoint_limit_01.json   limit   timeout       -1         0         0      0        0      600  0.000
multi_exec_01.json       multi   timeout       -1         0         0      0        0      600  0.000
stop_01.json             stop    timeout       -1         0         0      0        0      600  0.000
```

The completing scenario terminates normally with bounded context (`market_01`
peaked at 1,888 real prompt tokens). The other four degenerate into max-length
generation loops; the window keeps those histories at the 3K budget, but runs
are still slow (~3 tok/s on GTX 1650 Ti — a 12-step `stop_01` loop ran ~36 min
before being terminated). Full run:
`.venv/bin/python -m experiments.phase_10.run_real_after`.

Correctness is unchanged from phase 9 (0% report success at 0.6B — the model does
not construct the 24-column report); the orchestration layer is validated on
wiring, token consumption, and bounded context, not on the model's ability to
write the report.

## Commands

```bash
# controlled real 5-scenario batch (subprocess per scenario, 600 s cap)
.venv/bin/python -m experiments.phase_10.run_real_after

# single real scenario (no cap, trace saved on completion)
.venv/bin/python -m experiments.phase_10.run_real_after --run-one stop_01.json --max-steps 12

# offline before/after replay + wiring check
.venv/bin/python -m experiments.phase_10.validate_context

# benchmark CLI with strategy selection
.venv/bin/python -m finrl.benchmark --model qwen --mode real --prompt-strategy scenario_aware --limit 10
```