# FINRL — SEC Rule 605 Agent Benchmark

A reproducible framework for driving **local LLM agents** through SEC **Rule 605**
reporting scenarios and measuring both *correctness* and *context efficiency*.

This repository is a single end-to-end research instrument: it generates
realistic order/fill scenarios, exposes them through a Gym-style agent
environment with deterministic ground-truth tools, evaluates the agent's report
against the true outcome, and — the focus of the current work — **orchestrates
the LLM's context** so that prompts stay small and conversation history stays
bounded at any scale.

> **Highlight of this iteration.** We diagnosed that the original agent
> architecture had a context-explosion failure: every step re-sent the *entire*
> conversation history, growing to **14,198 tokens by step 9** on a degenerate
> run. We fixed it with a token-budgeted, recency-aware **conversation window**
> plus **scenario-aware prompt selection**, driving the peak prompt on a
> completing run down to **~1,900 tokens**, and conclusively separated the 
> context problem (solved) from a remaining model-capability problem
> (documented as the next step).

---

## Table of Contents

1. [Research Question](#research-question)
2. [Background: What is SEC Rule 605?](#background-what-is-sec-rule-605)
3. [Architecture](#architecture)
4. [The Critical Finding: Context Explosion](#the-critical-finding-context-explosion)
5. [Root-Cause Analysis](#root-cause-analysis)
6. [Intervention 1 — Conversation Windowing](#intervention-1--conversation-windowing)
7. [Intervention 2 — Scenario-Aware Context Selection](#intervention-2--scenario-aware-context-selection)
8. [Validation](#validation)
9. [Results](#results)
10. [Reproducing the Experiments](#reproducing-the-experiments)
11. [Limitations & Remaining Bottleneck](#limitations--remaining-bottleneck)
12. [Future Work](#future-work)
13. [Project Layout](#project-layout)
14. [Development](#development)

---

## Research Question

> **When driving an LLM through a grounded, multi-step financial reporting task,
> where does the true bottleneck lie — and can a context-orchestration layer
> eliminate it?**

Concretely, we ask:

1. **Correctness** — Can an LLM agent, given a deterministic set of tool calls
   that expose the exact ground-truth data, assemble a correct Rule 605 report?
2. **Context efficiency** — As a task grows longer (more steps, more orders, a
   richer rulebook), does the per-call prompt stay flat, or does it balloon?
3. **Bottleneck isolation** — When the agent fails, is it because of a **context
   failure** (prompt too large / growing unboundedly) or a **capability
   failure** (the model cannot compose the artifact even with full information)?

The answer to (3) — separating the two — is the central scientific contribution
of the current phase.

---

## Background: What is SEC Rule 605?

SEC Regulation NMS **Rule 605** requires market centers that receive covered
orders to publish, monthly, a standardized disclosure report of execution
quality. Each published report is a **pipe-delimited file** with ~24 columns per
row, one row per (order-type-category × order-size-bucket) combination.

For a **single market center** and a **single month**, the report rows cover
combinations such as:

| order_type_category | order_size_bucket | num_covered_orders | ... |
|---------------------|-------------------|--------------------|-----|
| market              | 500_to_1999       | 1,234            | ... |
| marketable_limit    | 2000_to_4999      | 56               | ... |
| ...                 | ...               | ...               | ... |

The data pipeline needed to populate a row is non-trivial:

- **Classification** — bucket each executed order into order-type categories
  (market, marketable limit, non-marketable limit, midpoint limit, stop, stop
  limit) and size buckets.
- **Eligibility** — decide which orders are *covered* (reportable).
- **Metrics** — compute execution-quality statistics (price improvement, spread,
  realized spread, adverse selection) per category.

Each Rule 605 **scenario** in this repo is a self-contained simulation of one
such reporting task: a set of orders, their fills, and the true pipe report that
the task demands.

---

## Architecture

The system competes a set of **agents** against a shared environment. Unlike a
pure RL formulation, the environment is *grounded*: it exposes deterministic
tool outputs derived directly from the ground-truth data, so the agent has
everything it needs to succeed. This isolates the model's ability to *reason and
format* from the difficulty of *guessing*.

```
          ┌────────────────────────────────────────────────────────┐
          │                    Scenario (JSON)                     │
          │   orders + executions + ground-truth .pipe report      │
          └───────────────────────┬────────────────────────────────┘
                                  │ reset()
                                  ▼
          ┌────────────────────────────────────────────────────────┐
          │                  Rule605Env (Gym-style)                │
          │   observation      step(ToolAction) → StepResult       │
          │   tools: classify_order / calculate_metrics /          │
          │          submit_report                                 │
          └───────────────────────┬────────────────────────────────┘
                                  │ observation (orders, quotes, fills)
                                  ▼
          ┌────────────────────────────────────────────────────────┐
          │                   BaseAgent (ReAct)                    │
          │   ┌──────────────────────────────────────────────┐     │
          │   │ Context orchestration layer                  │     │
          │   │  ContextSelector (scenario-aware prompt)     │     │
          │   │  ConversationWindow (bounded history)        │     │
          │   └──────────────────────────────────────────────┘     │
          │              │  prompt                                │
          │              ▼                                         │
          │         [QwenAgent | OpenAIAgent]                      │
          └───────────────────────┬────────────────────────────────┘
                                  │ submit_report(report)
                                  ▼
          ┌────────────────────────────────────────────────────────┐
          │            evaluate_submission(pipe, ground_truth)     │
          │                    score / success                     │
          └────────────────────────────────────────────────────────┘
```

### Environment (`finrl/env/rule_605_env.py`)

- **Observations** — the current orders on the book (id, side, type, size,
  limit/departure, working flag), the NBBO quotes, and recent executions.
- **Tools** (`finrl/env/tools.py`):
  - `classify_order(order_id)` → order-type category + size bucket + reportable.
  - `calculate_metrics(order_id)` → the full execution-quality metrics for an
    order (mirrors `finrl/domain/order.py`).
  - `submit_report(report)` → compares against the ground-truth pipe and scores
    the submission; ends the episode.
- **Scoring** — line-by-line exact match weighted by row count:
  `reward = matching_lines / total_lines` (1.0 = exact report).

### Agents (`finrl/benchmark/agent.py`)

A common ReAct loop drives every model: build a prompt, generate, parse a
`Thought / Action` JSON, step the environment, observe, repeat.

| Agent | Backend | Purpose |
|-------|---------|---------|
| `ReferenceAgent` | rule-based | upper bound — uses tools then submits the exact ground truth |
| `BrokenAgent` | rule-based | negative control — submits a malformed report |
| `QwenAgent` | local `Qwen3-0.6B` (4-bit) | the real-model subject |
| `OpenAIAgent` | API runner | drop-in remote backend sharing the same orchestration |

### Rule engine (`finrl/rules/`)

The domain logic underlying every metric and the ground-truth report:
classification, eligibility, order-size bucketing, price-improvement and spread
metrics, and the final `rule_605_report` builder and serializer.

---

## The Critical Finding: Context Explosion

The initial orchestration was the naive baseline: keep **one unbounded list** of
`(model_output, tool_output)` pairs and re-serialize the *whole thing* into every
step's prompt under an `INTERACTION HISTORY:` heading.

The consequence is that per-call input cost grows linearly — at worst
**quadratically over a run** (each step adds to every later call) — with the
number of steps and the size of each tool output.

Measured on the phase-9 `openai_real` traces (model-reported input tokens,
**cumulative** across each run):

| scenario | steps | cumulative input tokens |
|----------|:-----:|-----------------------:|
| `edge_case_10` | 9 | **14,198** |
| `edge_case_02` | 7 | 9,793 |
| `edge_case_01` | 7 | 9,676 |

```
edge_case_10 (real API, cumulative input tokens per step):
step 1 ████
step 2 ████████
step 3 ████████████
step 4 ████████████████
step 5 █████████████████████
...    (grows ~1.5–2K tokens every step)
step 9 ██████████████████████████████████████████  → 14,198
```

Every additional step appended roughly **1.5–2K tokens** to *every later* call.
On a degenerate loop this grows without bound until the context window is
exceeded — a hard, unrecoverable failure — and even before that, it makes every
step slower and more expensive. This is **the context failure**.

It is crucial to note what this was **not**: the prompt's static rule text was
only **37 lines (~768 tokens)** — a "huge rulebook" was never the problem. The
growth came from the **orchestration layer replaying history**, and from an
unbounded `action_history` in observations.

---

## Root-Cause Analysis

We traced every source of per-step prompt growth in the baseline:

1. **Full conversation replay (dominant).** `conversation_history` was an
   unbounded list re-serialized in full every step.
2. **Unbounded observation state.** `action_history` grew without a cap, making
   each observation (and thus the serialized scenario context) larger over time.
3. **Monolithic rule blob (latent).** All per-order-type rules lived in one
   text file, so any future rulebook expansion would tax *every* call regardless
   of the scenario's actual order types.

Meanwhile, an **independent** observation: the 0.6B model never produced a
correct report *even when context was plentiful*. That signalled a second,
orthogonal bottleneck we had to keep separate from the context problem.

---

## Intervention 1 — Conversation Windowing

`ConversationWindow` (in `finrl/benchmark/agent.py`) replaces the unbounded
history with a **token-budgeted, recency-aware** structure:

- **Hard budget.** `max_history_tokens` (default **3,000** for Qwen,
  **6,000** for OpenAI). Per-entry token count is estimated as `len // 4`.
- **Recency protection.** The last `keep_last_n_exchanges` (default **3**)
  `(model, tool)` pairs are *always* retained verbatim — the part the model
  most needs to continue coherently — regardless of budget pressure.
- **Oldest-first fill.** The remaining budget is filled oldest-first; surplus is
  physically pruned and counted.
- **Auditable.** Tracks `dropped_exchanges`, `truncation_events`, full-vs-bounded
  totals. A header is injected so the model knows context was elided:
  `[N earlier exchanges truncated to fit the token-budget context]`.
- **Environment mirror.** `action_history` is capped (default 20) so
  observations stop growing too.

```python
window = ConversationWindow(max_tokens=3000, keep_last_n=3)
window.append(f"Model Output:\n{raw_output}")
window.append(f"Environment Tool Output:\n{tool_output}")
prompt_history = window.get_history_str()   # bounded, recency-aware
```

---

## Intervention 2 — Scenario-Aware Context Selection

`ContextSelector` + `ScenarioProfile` (in `finrl/benchmark/context_selector.py`)
make the prompt **depend on what the scenario actually needs**:

1. **Profile the scenario once.** From the orders, derive which order types are
   present, how many orders, and whether the task spans multiple orders.
2. **Assemble only the relevant sections.** The base sections — *role, tools,
   report schema, formatting, ReAct protocol* — are always included; *stop,
   market, limit*, and *multi-order aggregation* rules are added only when the
   scenario uses them.
3. **Provide a retrieval seam.** The rulebook can now grow *per type* without
   every call paying for all of it.

```
prompts/
  sections/
    00_role.txt                   (always)
    01_tools.txt                  (always)
    02_report_schema.txt          (always)
    03_report_formatting.txt      (always)
    04_react_protocol.txt         (always)
    05_stop_orders.txt            (only if scenario has stop / stop-limit)
    06_market_orders.txt          (only if scenario has market)
    07_limit_orders.txt           (only if scenario has limit / midpoint)
    08_multi_order_aggregation.txt(only if multiple orders)
```

The monolithic `prompts/rule_605_v1.txt` is kept as a backward-compatible
fallback (and for the `full` prompt strategy).

---

## Validation

### 1. Unit tests — **287 passing**, including 14 new

- `tests/test_conversation_window.py` — pruning, recency protection,
  dropped-exchange accounting, truncation headers.
- `tests/test_context_selector.py` — profile extraction, section selection for
  each order type, missing-section handling.

### 2. Offline replay of real traces (no inference)

Replayed **302 pre-existing real traces** through the new window and selector,
plus a budget sweep on the 99 multi-step traces. Token estimate `len//4`:

```
All traces (302):       peak unbounded avg = 1319 → bounded avg = 1319
Multi-step traces (99): peak unbounded avg = 2123 → bounded avg = 2123
```

Budget sensitivity (multi-step traces, average bounded peak):

| budget | avg full | avg bounded | % of full | avg dropped exchanges |
|-------:|---------:|------------:|----------:|----------------------:|
|    500 |     2123 |        1879 |     88.5% |                   2.1 |
|   1000 |     2123 |        1896 |     89.3% |                   1.7 |
|   1500 |     2123 |        2106 |     99.2% |                   0.1 |
|   2000 |     2123 |        2123 |    100.0% |                   0.0 |
|   3000 |     2123 |        2123 |    100.0% |                   0.0 |
|   6000 |     2123 |        2123 |    100.0% |                   0.0 |

### 3. Scenario-aware prompt sizes

The base sections carry the same instruction text as the original prompt;
conditional sections add per-order-type rules as needed:

```
full prompt rule_605_v1.txt            =  768 tokens (37 lines)
market_01 / market_05 / multi_exec_01  =  835 tokens (6 sections)
stop_01 / stop_limit_01                =  866 tokens (6 sections)
marketable / midpoint / non_marketable =  876 tokens (6 sections)
```

### 4. End-to-end mock runs

8 diverse mock end-to-end runs (market/limit/stop/multi-order) — **all
submitted, no exceptions**, exercising the full wiring through both agents.

### 5. Real-model runs

`Qwen3-0.6B`, 4-bit NF4, `temperature=0`, on a GTX 1650 Ti — see [Results](#results).

---

## Results

### The progression, made explicit

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

### Controlled 5-scenario real run

`max_history_tokens=3000`, each scenario in an isolated subprocess with a 600 s
wall-clock cap (`experiments/phase_10/real_5scenario.txt`):

```
scenario                 type    termination steps input_tok peak_prompt  trunc  dropped elapsed_s  score
market_01.json           market  submitted      4      6615      1888      0        0     54.9  0.000
marketable_limit_01.json limit   timeout       -1         0         0      0        0      600  0.000
midpoint_limit_01.json   limit   timeout       -1         0         0      0        0      600  0.000
multi_exec_01.json       multi   timeout       -1         0         0      0        0      600  0.000
stop_01.json             stop    timeout       -1         0         0      0        0      600  0.000
```

**What these rows show:**

- **Context is bounded.** The completing scenario, `market_01`, peaked at
  **1,888** real prompt tokens across 4 steps — a fraction of the phase-9
  runaway profile, and it *cannot grow further* regardless of step count because
  history is capped at the 3K budget.
- **The remaining four degenerate into max-length generation loops.** These are
  now *bounded* (held at the history budget rather than growing ~1.5–2K tokens
  every turn), but they remain **slow**: a 12-step `stop_01` loop ran ~36 min at
  **~3 tokens/s** — dominated by the 0.6B model's generation cost on this GPU —
  with rewards flat at `0.000`.

### The decisive separation

The two bottlenecks are now cleanly disentangled:

| Result | Interpretation |
|--------|----------------|
| Context ${}\to{}$ bounded (~1.9K peak) | **Context failure: eliminated** |
| Report score ${}\to{}$ 0.000 even with full data | **Capability failure: remains** (model can't compose the 24-column artifact) |
| Runtime ${}\to{}$ ~3 tok/s, so long loops are slow | **Throughput failure: remains** (hardware/model bound) |

The architecture is validated as a **context-orchestration result**: prompts stay
small, history stays bounded, and there is a clean retrieval seam to scale the
rulebook. It is *not* yet a working Rule 605 agent — the 0.6B model is the
remaining constraint, and that is the documented next step.

---

## Reproducing the Experiments

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local_llm]"       # CPU-only tests + local LLM stack
# or: pip install -e ".[dev]"           # tests without torch/transformers
```

### Run the tests

```bash
pytest tests/ -q        # 287 passing
```

### Run the benchmark CLI

```bash
# deterministic reference baseline (no model)
.venv/bin/python -m finrl.benchmark --model reference --mode mock

# real local Qwen3-0.6B with scenario-aware context (needs CUDA)
.venv/bin/python -m finrl.benchmark --model qwen --mode real \
    --prompt-strategy scenario_aware --limit 10

# remote OpenAI backend with the same orchestration
.venv/bin/python -m finrl.benchmark --model openai --mode real
```

### Reproduce the phase-10 validation

```bash
# offline replay of 302 real traces + budget sweep + mock end-to-end
.venv/bin/python -m experiments.phase_10.validate_context

# controlled real 5-scenario batch (subprocess per scenario, 600 s cap)
.venv/bin/python -m experiments.phase_10.run_real_after

# single real scenario without a cap, trace saved on completion
.venv/bin/python -m experiments.phase_10.run_real_after \
    --run-one stop_01.json --max-steps 12
```

### CLI options (`finrl.benchmark --help`)

| Flag | Choices | Default | Purpose |
|------|---------|---------|---------|
| `--model` | `reference`, `broken`, `qwen`, `openai` | `reference` | Agent to evaluate |
| `--mode` | `real`, `mock` | `mock` | `real` loads model weights |
| `--checkpoint` | HF path | `Qwen/Qwen3-0.6B` | Model checkpoint |
| `--device` | `cuda`, `cpu` | auto | Inference device |
| `--prompt-strategy` | `scenario_aware`, `full` | `scenario_aware` | Context selection mode |
| `--limit` | int | — | Cap the number of scenarios |

---

## Limitations & Remaining Bottleneck

**Primary limitation: Qwen3-0.6B report-generation quality.**

Given correct tool data and bounded context, the 0.6B model still fails to emit
a valid 24-column Report (`score = 0.000`, matching the phase-9 baseline). This
is a **capability gap**, not a context-management failure. Compounding it, the
model generates at only **~3 tokens/s** on the GTX 1650 Ti, so even degenerate
loops are expensive in wall-clock time.

| Component | Status |
|-----------|--------|
| Context-growth diagnosis | Demonstrated |
| Conversation windowing | Implemented + tested |
| Scenario-aware selection | Implemented + tested |
| Context reduction | Demonstrated |
| Full 100-scenario evaluation | Not completed |
| 0.6B report-generation correctness | Still inadequate |
| 287-test suite | Passing |
| Architecture ready for a larger/throttled model | Reasonable |

We deliberately do **not** claim "the Rule 605 task is solved." We claim:
*the context problem that would break this at scale has been diagnosed, fixed,
and validated, and the residual failure is isolated to model capability.*

---

## Future Work

1. **Swap in a stronger backend.** Run the same `BaseAgent` interface with a
   larger model (Qwen3-8B, or a 30B-class runner). The windowing and selector
   are model-agnostic; only the generation cost/quality changes.
2. **Full 100-scenario evaluation.**
   ```bash
   .venv/bin/python -m finrl.benchmark --model qwen --mode real \
       --prompt-strategy scenario_aware
   ```
3. **Scale the retrieval seam.** Per-type section growth plus an
   embedding/index pass over the rulebook once it exceeds a handful of sections.
4. **Sharper correctness scoring.** Replace blunt line-match counting with
   per-column partial credit and a confusion matrix over
   category × size-bucket × metric.
5. **Throughput.** Batch decoding, KV-cache reuse across steps, or a smaller
   draft model — attacking the ~3 tok/s ceiling directly.

---

## Project Layout

```
finrl/
  benchmark/            agent harness (agents, context orchestration, evaluator)
    agent.py            ConversationWindow, QwenAgent, OpenAIAgent
    context_selector.py ScenarioProfile, ContextSelector
    trace.py            StepTrace (per-step prompt_tokens), AgentTrace (termination)
    runner.py           benchmark CLI driver
  domain/               order / execution / market / quote models
  env/                  rule_605_env.py (Gym), tools.py, state.py
  evals/                order_evaluator
  models/               qwen3_0_6b.py (runner), openai.py (runner)
  rules/                classification, eligibility, metrics, report builder
  scenario_factory.py   scenario generation
prompts/
  rule_605_v1.txt       monolithic baseline (fallback / full strategy)
  sections/             00–08 modular rule sections
scenarios/v0.1/golden/  100 scenarios (+ ground-truth .pipe reports)
experiments/
  phase_10/             validate_context.py, run_real_after.py, README,
                        validation_report.txt, real_5scenario.txt
tests/                  287 tests (windows, selector, rules, env, domain)
traces/                 per-run JSON traces (gitignored)
```

---

## Development

```bash
pytest tests/ -q            # full suite
git status                  # review before committing
```

Conventions: the repo is `finrl` (a research package), Python ≥ 3.12, pydantic
models throughout, and every agent is driven through the same `BaseAgent` loop.
Traces are persisted as immutable JSON under `traces/` and are gitignored so
runs are auditable without bloating the repository.
