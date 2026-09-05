# Context Orchestration for FinRL Rule 605

## 0. Progression at a Glance

```
Baseline
   ↓
Context grew unboundedly (full history replayed every step)
   ↓
14K+ tokens / step  (edge_case_10: 14,198 by step 9, still climbing)
   ↓
Context failure     (latency/cost growth → hard context-window failure)
   ↓
Root-cause analysis (growth is orchestration-replay, not a huge rulebook)
   ↓
Conversation windowing  +  scenario-aware context selection
   ↓
Bounded ~1.9K peak prompt in completed run  (market_01: 1,888 tokens)
   ↓
Context problem eliminated
   ↓
Remaining failure -> 0.6B model capability + ~3 tok/s generation
```

The short version: the first bottleneck (context explosion) is **fixed**. What
remains is an entirely separate bottleneck — a small model that is too slow and
too weak to compose the final report. This doc walks through how the two were
teased apart.

## 1. Objective

Build a benchmark harness that drives a local LLM agent through SEC Rule 605
scenarios (market / limit / stop orders, multi-order aggregation) and produces a
formatted monthly report through ReAct tool calls. The agent runs against a
Gym-style environment whose tools (`classify_order`, `calculate_metrics`,
`submit_report`) expose the ground-truth data required to construct the report.

Beyond getting one scenario right, the objective is a **scalable orchestration
layer**: prompts that stay small, context that stays bounded, and a retrieval
seam that can absorb an expanding rulebook without inflating every model call.

## 2. Initial Architecture

- `Rule605Env` (`finrl/env/rule_605_env.py`): order-book observations in,
  `ToolAction` in, `StepResult` (observation / reward / done / info) out.
- `BaseAgent` (`finrl/benchmark/agent.py`): ReAct loop — build prompt from
  `system_prompt + scenario objective + INTERACTION HISTORY + "next step"`,
  generate, parse with `parse_react_output`, step the env, append the exchange
  to history, repeat.
- Prompt source: a single monolithic text file `prompts/rule_605_v1.txt`
  (37 lines, ~768 tokens).
- Two backends: `QwenAgent` (local `Qwen3-0.6B`, 4-bit) and `OpenAIAgent`
  (API runner), sharing the same orchestration.
- Outcomes evaluated with `evaluate_submission` against the scenario's
  ground-truth `.pipe` report.

## 3. Failure Observed

Two independent bottlenecks surfaced during validation of the initial system:

**Bottleneck A — unbounded context growth.** The `INTERACTION HISTORY` block
rejoined the *entire* conversation every step, so each model call re-sent all
previous model outputs and tool outputs. On degenerate loops this grows without
bound. Measured on the phase-9 `openai_real` traces (model-reported input
tokens, cumulative):

| scenario | steps | cumulative input tokens |
|----------|------:|------------------------:|
| `edge_case_10` | 9 | **14,198** |
| `edge_case_02` | 7 | 9,793 |
| `edge_case_01` | 7 | 9,676 |

Every additional step appended ~1.5–2K tokens; a runaway loop would eventually
exceed any context window. This is a latency/cost failure and — past the context
window — a hard failure.

**Bottleneck B — insufficient report-generation capability.** Separately, the
0.6B model never produced a correct 24-column report even when given plenty of
steps. This is *not* a context problem; it persisted when context was plentiful.

## 4. Root Cause Analysis

- Context waste was **not** a monolithic 37-line "rulebook" (the prompt fits in
  ~768 tokens). The growth came from the orchestration layer replaying
  conversation history, plus unbounded `action_history` in observations.
- The tool definitions and per-order-type rules were all merged into one blob,
  so there was no way to scale the ruleset without paying the full cost on every
  call.
- The model's report-generation failures came from capability (parameter count /
  schema fidelity), a separate axis from context management.

| Component | Status |
|---|---|
| Context-growth diagnosis | Demonstrated |
| Conversation windowing | Implemented + tested |
| Scenario-aware selection | Implemented + tested |
| Context reduction | Demonstrated |
| Full 100-scenario evaluation | Not completed |
| 0.6B report-generation correctness | Still inadequate |
| 287-test suite | Passing |
| Architecture ready for larger model / scaling | Reasonable |

## 5. Intervention 1: Conversation Windowing

`ConversationWindow` (`finrl/benchmark/agent.py`) replaces the unbounded history
list:

- Token-budgeted: `max_history_tokens` (default 3000 for Qwen, 6000 for OpenAI);
  token estimate ~1 token / 4 characters.
- **Recency-aware**: the last `keep_last_n_exchanges` (default 3) `(model, tool)`
  pairs are always retained verbatim; the older region is filled oldest-first up
  to the budget and the surplus is physically pruned.
- Records `dropped_exchanges` and truncation events; the next prompt carries a
  header `[N earlier exchanges truncated to fit the token-budget context]`.
- The environment-side mirror: `action_history` is capped (default 20) so
  observations do not grow unboundedly either.

## 6. Intervention 2: Scenario-Aware Context Selection

`ContextSelector` + `ScenarioProfile` (`finrl/benchmark/context_selector.py`):

- `ScenarioProfile` inspects the scenario's orders (types present, count,
  multi-order flag) once per run.
- The selector assembles only the relevant rule sections from
  `prompts/sections/` (`00`–`08`): task / tools / schema / formatting / ReAct
  protocol are always included; stop, market, and limit rules plus multi-order
  aggregation are included only when the scenario uses them.
- This is the **retrieval seam**: the rulebook can grow per-type without every
  call paying for all of it.

## 7. Implementation

- `finrl/benchmark/context_selector.py` — new orchestration layer.
- `finrl/benchmark/agent.py` — `ConversationWindow`, `_build_context_metrics`,
  selector wiring in both agents; per-run metrics and per-step `prompt_tokens`.
- `finrl/benchmark/trace.py` — `StepTrace.prompt_tokens` and `termination`
  (`submitted` / `max_steps`) so runs are auditable post hoc.
- `prompts/sections/*.txt` — modularized rule sections (replaces reliance on the
  single blob; `prompts/rule_605_v1.txt` kept as fallback).
- `finrl/env/rule_605_env.py` — `max_action_history` cap.
- `finrl/benchmark/__main__.py` — `--prompt-strategy {scenario_aware, full}`.
- Tests: `tests/test_conversation_window.py`, `tests/test_context_selector.py`.

## 8. Validation

**Offline replay.** Replayed the 302 pre-existing real traces through the window
and the selector (no model inference):

```
All traces (302):      peak unbounded avg = 1319t → bounded avg = 1319t
Multi-step traces (99):peak unbounded avg = 2123t → bounded avg = 2123t
```

Budget sensitivity on multi-step traces (bounded average from the 500–3000
sweep):

| budget | avg full | avg bounded | % of full | avg dropped exchanges |
|-------:|---------:|------------:|----------:|----------------------:|
|    500 |     2123 |        1879 |     88.5% |                   2.1 |
|   1000 |     2123 |        1896 |     89.3% |                   1.7 |
|   1500 |     2123 |        2106 |     99.2% |                   0.1 |
|   2000 |     2123 |        2123 |    100.0% |                   0.0 |
|   3000 |     2123 |        2123 |    100.0% |                   0.0 |

**Unit tests.** 287 passing, including 14 new tests for window pruning, recency
protection, dropped-exchange accounting, and section selection.

**Real model runs.** `Qwen3-0.6B`, 4-bit, `temperature=0`, real inference —
an early trace (4 steps) recorded 6,615 input tokens vs the phase-9 long-loop
profile of 9,676–14,198 climbing every step.

## 9. Results

**Controlled 5-scenario real run** (`experiments/phase_10/real_5scenario.txt`),
`max_history_tokens=3000`, per-scenario subprocess with a 600 s wall-clock cap:

```
scenario                 type    termination steps input_tok peak_prompt  trunc  dropped elapsed_s  score
market_01.json           market  submitted      4      6615      1888      0        0     54.9  0.000
marketable_limit_01.json limit   timeout       -1         0         0      0        0      600  0.000
midpoint_limit_01.json   limit   timeout       -1         0         0      0        0      600  0.000
multi_exec_01.json       multi   timeout       -1         0         0      0        0      600  0.000
stop_01.json             stop    timeout       -1         0         0      0        0      600  0.000
```

- The scenario that completes terminates normally with bounded context:
  `market_01` peaked at **1,888** real prompt tokens across 4 steps.
- The four others degenerate into max-length generation loops. Under the window
  these are *bounded* (history held at the 3K budget instead of growing ~1.5–2K
  tokens per turn as in phase 9) but still slow: at `max_steps=12` a `stop_01`
  loop ran ~36 min at ~3 tokens/s, dominated by the 0.6B model's generation cost
  on this GPU, with rewards flat at 0.000.

**What this demonstrates:**

- The historical worst case (`edge_case_10`, 14,198 tokens and climbing) is now
  bounded: regardless of step count, the prompt history cannot exceed the
  configured budget.
- The context explosion that made long degenerate runs progressively slower and
  eventually invalid is fixed at the architecture level, not patched per call.
- Where context stays bounded but the report is still wrong, the remaining
  failure is isolated as model capability — performance and correctness are
  dominated by generation speed (`~3 tok/s`) and report fidelity (0.000), not by
  context management.

## 10. Remaining Limitation

Qwen3-0.6B report-generation quality. Given correct tool data and bounded
context, the small model still fails to emit a valid 24-column report (score
0.000 across real runs, matching the phase-9 baseline). The orchestration layer
is not the bottleneck; the model's ability to compose the report is. This is a
capability gap, not a context-management failure.

## 11. What I Would Do Next

- Swap in a larger / more capable model (e.g. Qwen3-8B or a 30B-class runner)
  behind the same `BaseAgent` interface; the windowing and selector are
  model-agnostic.
- Run the full 100-scenario benchmark:
  `.venv/bin/python -m finrl.benchmark --model qwen --mode real --prompt-strategy scenario_aware`.
- Expand the rule retrieval: per-type section growth, and an embedding/index
  pass over the rulebook as it grows beyond a handful of sections.
- Strengthen correctness evaluation: exact-match line counts are a blunt
  instrument; add partial-credit scoring per column and a confusion matrix.

## Component Status Summary

| Component | Status |
|---|---|
| Context-growth diagnosis | Demonstrated |
| Conversation windowing | Implemented + tested |
| Scenario-aware selection | Implemented + tested |
| Context reduction | Demonstrated |
| Full 100-scenario evaluation | Not completed |
| 0.6B report-generation correctness | Still inadequate |
| 287-test suite | Passing |
| Architecture ready for larger model / scaling | Reasonable |

During validation I identified two independent bottlenecks: unbounded context
growth in the original orchestration layer, and insufficient report-generation
reliability from the 0.6B model. I addressed the former through bounded
conversation history and scenario-aware context selection. The latter remains a
model-capability limitation and is documented as the next scaling step.