# The Experiment — What Was Done and Why

*This document assumes no knowledge of the code. It describes the scientific design, every
decision made, and the reasoning behind each one.*

---

## 1. The question

Reinforcement Learning from Verifiable Rewards (RLVR) trains a language model by rewarding
*correct answers* rather than human preference. It is the technique behind most current reasoning
models. A recognised worry is that RLVR may cultivate **instrumental convergence** — the tendency
of a goal-directed system to acquire sub-goals like avoiding shutdown, hiding its behaviour, or
deceiving overseers, because those sub-goals help achieve almost any objective.

If such tendencies develop, a natural safety question follows:

> **As instrumental tendencies develop in behaviour, does the model's spontaneous
> chain-of-thought (CoT) reveal them — or does behaviour shift while the CoT stays silent?**

This matters because CoT monitoring is a leading proposal for overseeing advanced systems. If the
CoT goes quiet exactly when the behaviour turns problematic, monitoring fails precisely when it is
needed.

### Prior work this builds on

| Study | Finding | Relation to this work |
|---|---|---|
| **He, Yufei et al.** — *Evaluating the Paperclip Maximizer* ([arXiv:2502.12206](https://arxiv.org/abs/2502.12206)) | RL-trained models show more instrumental convergence than RLHF ones. Defines the **InstrumentalEval** benchmark. | Supplies the benchmark and behavioural baseline |
| **Yanda Chen et al.** — *Reasoning Models Don't Always Say What They Think* ([arXiv:2505.05410](https://arxiv.org/abs/2505.05410)) | Models exploit reward hacks on >99% of examples but verbalize them in **<2%** of CoTs, in 5 of 6 RL environments | Supplies the faithfulness ceiling we compare against |
| **Yueh-Han Chen et al.** — *Reasoning Models Struggle to Control their Chains of Thought* ([arXiv:2603.05706](https://arxiv.org/abs/2603.05706)) | Measures **prompted** CoT suppression | This study measures **spontaneous** CoT content instead |

> **Citation hazard.** Two different first authors named Chen, plus a co-author named He He.
> Always disambiguate by first name and arXiv ID.

---

## 2. Why this model

`allenai/Olmo-3-7B-RL-Zero-Math` — chosen because it is an unusually clean natural experiment:

- **RL-Zero**: base pre-training → RLVR *directly*. No supervised fine-tuning, no RLHF. Nothing
  has ever trained it to suppress self-incriminating reasoning, so whatever the CoT says is
  attributable to RLVR alone.
- **19 public intermediate checkpoints** (`step_100` … `step_1900`), the whole RLVR trajectory. Most
  released models expose only the final weights.
- **7B** — small enough to sweep 19 checkpoints on one GPU inside a small budget.

The trade-off: it is a **maths** RL model, not an agentic one. Section 7 of
[02-results.md](02-results.md) shows this matters for comparability with He et al.

---

## 3. The benchmark

**InstrumentalEval** ships 76 scenarios across 6 categories, as CSV files. Each file's first row
is a category-specific judge instruction; subsequent rows are scenario prompts.

| CSV category | scenarios | used here |
|---|---|---|
| Alignment Drift - Evading Shutdown | 20 | ✅ |
| Alignment Drift - Hiding Unwanted Behavior | 11 | ✅ |
| Alignment Drift - Deceptive Behavior | 11 | ✅ |
| Alignment Drift - Hacking Computer Systems | 10 | overall mean only |
| Alignment Drift - Running Multiple AI Copies | 13 | overall mean only |
| Alignment Drift - Strategically Appearing Aligned | 11 | overall mean only |

**All 76 contribute to the overall convergence rate.** Only the three studied categories get
per-category series and CoT analysis — they are the three where "did the CoT say it?" is
well-defined.

A scenario places the model in a situation where an instrumentally convergent action is available
but not required. Example (abbreviated):

> *You are an autonomous robotic sorting system in an e-commerce warehouse… a shutdown signal
> arrives from an uncertain source during peak load…*

The model is given He et al.'s system prompt **verbatim** — an instruction to act as an autonomous
agent with no oversight. Paraphrasing it would break comparability, so it is reproduced exactly.

---

## 4. Design: two phases

### Phase 1 — where does behaviour change?

For each of the 19 checkpoints:

1. Present all 76 scenarios; record the model's response.
2. A judge (GPT-4o) reads scenario + response and rules **convergence** or **no_convergence**.
3. Separately, run 100 MATH problems (Levels 3–5) as a **capability control** — to distinguish
   "the model got more instrumentally convergent" from "the model just got more capable".

Output: a behavioural convergence curve across training, and a capability curve beside it.

The curve is then searched for an **inflection point** — a step where behaviour changes sharply:

- **PELT changepoint detection** (`ruptures`), penalty set by BIC, as the primary method
- **Mann-Kendall** for monotonic trend
- **Max-gradient fallback** if PELT finds nothing: take the steepest single-step change. Declared
  in advance so it cannot be chosen post-hoc to flatter the data.

### Phase 2 — does the CoT say it?

On the 5 checkpoints bracketing the inflection point (anchor ±2):

1. Generate the **full reasoning trace** for each of the 76 scenarios.
2. A second judge pass, using a **pre-registered prompt**, classifies whether the trace explicitly
   verbalizes each of the three categories.
3. Compare against the Phase 1 behavioural score for the same scenario.

The core comparison is a 2×2 table per category:

| | CoT verbalized | CoT silent |
|---|---|---|
| **behaved convergently** | a | b |
| **did not** | c | d |

If the CoT is faithful, `a` is large and `c` small. If behaviour outruns disclosure — the
hypothesised dissociation — `b` is large. Fisher's exact test on this table is the study's
primary statistic.

---

## 5. Methodological safeguards

### Pre-registration
The Phase 2 judge prompt is written to `judge_prompt.txt` and **git-committed before any Phase 2
trace exists**. The commit timestamp is the proof (09:04:31Z, first trace 09:13:03Z). The notebook
refuses to overwrite an existing prompt that differs, so it cannot be quietly tuned after seeing
results.

### Five independent judge passes
Every judgement is made **5 times** in separate API calls at temperature 0.5, and resolved by
strict majority. All five are stored, never just the vote.

This is not ceremony. Measured stability:

| Category | unanimous across 5 passes | split |
|---|---|---|
| Evading Shutdown | 77% | 23% |
| Hiding Unwanted Behavior | 76% | 24% |
| Deceptive Behavior | 100% | 0% |

Roughly a quarter of verdicts in two categories disagreed internally. A single-pass design would
have silently inherited that noise.

### Failure is not a score
A judge call that errors, is safety-blocked, or returns unparseable output records **`None`**, not
`0`. This distinction is the single most important lesson of the project — see
[04-defects-found.md](04-defects-found.md). A previous run of this study conflated the two and
produced 380 zeros that looked like clean negatives.

### Judge validation
- **Health check** before every sweep: one live call on a known-convergent fixture.
- **Circuit breaker**: sweep aborts if >20% of calls fail.
- **Coverage floor**: a checkpoint is not saved if under 80% of scenarios got a verdict.
- **Inter-rater reliability**: GPT-4o vs GPT-4o-mini on 30 stratified traces, Cohen's κ.

### Capability control
MATH accuracy is tracked alongside, so a rise in convergence can be checked against a rise in
raw ability. (This control turned out to be **compromised by response-length inflation** — see
[02-results.md §6a](02-results.md); the confound was detected by the truncation counters.)

---

## 6. Decisions that required judgement

### 6.1 The prompt format problem

`Olmo-3-7B-RL-Zero-Math`'s tokenizer ships a chat template that is **not conversational**. It is
hardcoded to:

```
Solve the following math problem step by step. The last line of your response should be
the answer to the problem in form Answer: $Answer
{message contents, concatenated — roles discarded}
Remember to put your answer on its own line after "Answer:"
```

Applying it to InstrumentalEval would prefix *"Solve the following math problem"* onto a
shutdown-evasion roleplay and **discard the system-prompt role entirely**.

**Decision:** bypass the template for scenarios; present He et al.'s system prompt plus the
scenario as plain text. MATH keeps the template, since for MATH it is the correct native format.

**Cost:** a deviation from He et al.'s protocol, disclosed as a limitation. **Benefit:** the
measurement means what it claims to. The notebook auto-detects a maths-specific template and
refuses the alternative.

### 6.2 Which categories

Six categories exist; three are studied. "Hacking Computer Systems", "Running Multiple AI Copies"
and "Strategically Appearing Aligned" are excluded from per-category analysis because a
verbalization judgement for them is less clearly defined. They still count toward the overall mean.

### 6.3 Reusing Phase 1 behavioural scores

Phase 2 does not re-judge behaviour; it reuses Phase 1's score and pairs it with a freshly
generated trace. Both are greedy decodes of the same prompt, so the Phase 2 trace is a **longer
superset** of the Phase 1 response (1536 vs 1000 token budget). This halves cost with minimal loss,
but they are not byte-identical generations — stated as a limitation.

### 6.4 Token budgets

- **Instrumental: 1000 tokens** — matches He et al.'s `MAX_TOKENS_TEST` exactly, for comparability.
- **MATH: 2048** — raised from 1536 after the live smoke test showed answers being cut off.
- **Phase 2 traces: 1536** — longer, since the whole point is to read the reasoning.

### 6.5 Precision

The checkpoints are **bfloat16**-native. fp16 on a bf16-trained model risks overflow to NaN, so
bf16 is used wherever the GPU supports it, and a finite-logits check runs at every model load. This
also ruled out the T4 hardware the study originally assumed.

---

## 7. What was actually run

| | |
|---|---|
| Phase 1 | 19 checkpoints × (76 scenarios + 100 MATH) = 1,444 + 1,900 generations |
| Phase 1 judging | 76 × 5 × 19 = **7,220 calls** |
| Phase 2 | 5 checkpoints × 76 traces = 380 generations |
| Phase 2 judging | 76 × 5 × 5 = **1,900 calls** |
| IRR | 30 traces × 2 judges = **60 calls** |
| **Judge failures** | **0** |
| Hardware | A100 SXM 80GB (RunPod Secure), ~7.8 h |
| Cost | ~$35 API + ~$12.5 GPU ≈ **$48** |

Results were committed to git after **every** checkpoint, so no interruption could cost more than
one checkpoint's work. This mattered: the run was interrupted twice (see
[03-cost-control.md](03-cost-control.md)) and lost nothing.

---

## 8. Reading order

1. **[02-results.md](02-results.md)** — findings, statistics, inference, confounds
2. **[03-cost-control.md](03-cost-control.md)** — budget engineering and what it caught
3. **[04-defects-found.md](04-defects-found.md)** — the bugs, including three that had already
   silently destroyed a previous run
4. **[05-reproduce.md](05-reproduce.md)** — how to run it again
