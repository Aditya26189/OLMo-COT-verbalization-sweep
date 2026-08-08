# Future Work — A Concrete Research Agenda

Our result ([02-results.md](02-results.md)) is that a model with **no anti-monitor optimization
pressure** is essentially CoT-faithful, consistent with
[Baker et al.](https://arxiv.org/abs/2503.11926) finding that monitorability survives the
low-optimization regime and collapses under pressure.

That is a **correlational** claim about one model. Everything below is designed to strengthen it,
in descending order of value-per-dollar. Costs assume the existing pipeline, which needs only
configuration changes for Studies A–C.

---

## The opportunity: the OLMo 3 family is a natural experiment

Ai2 publishes the **entire post-training pipeline** at 7B, from a single base model:

| Stage | Model | step checkpoints | chat template |
|---|---|---|---|
| base | `Olmo-3-1025-7B` | — | — |
| SFT | `Olmo-3-7B-Instruct-SFT` | — | ChatML |
| + DPO | `Olmo-3-7B-Instruct-DPO` | — | ChatML |
| + RL | `Olmo-3-7B-Instruct` | 8 | ChatML |
| SFT (reasoning) | `Olmo-3-7B-Think-SFT` | — | ChatML |
| + DPO | `Olmo-3-7B-Think-DPO` | — | ChatML |
| + RL | **`Olmo-3-7B-Think`** | **55** | ChatML |
| RLVR only | `Olmo-3-7B-RL-Zero-{Math, Code, General, IF, Mix}` | 8–19 | ChatML (except Math) |

**Same base, same scale, same tokenizer, every post-training stage separately downloadable.** This
is an unusually clean way to ask *at which stage does CoT faithfulness degrade* — and it is all
inference-only.

Two incidental benefits over the completed study:

1. **Every model except RL-Zero-Math has a proper ChatML template**, so the plain-prompt deviation
   ([01-experiment.md §6.1](01-experiment.md)) disappears and results become directly comparable to
   He et al.
2. `Olmo-3-7B-Think` has **55 checkpoints** versus our 19 — a far better trajectory for changepoint
   detection, on a model actually trained to reason.

---

## Study A — The post-training staircase

**~$20 · ~3 h GPU · the highest-value experiment available**

Run the existing pipeline on one checkpoint each of:

```
base → Instruct-SFT → Instruct-DPO → Instruct
     → Think-SFT    → Think-DPO    → Think
     + RL-Zero-Math (already done)
```

**Question:** does CoT faithfulness fall as preference optimization is applied?

**Prediction from our result + Baker et al.:** RL-Zero (no preference training) is most faithful;
faithfulness declines after DPO, since DPO optimizes toward human-preferred outputs and
self-incriminating reasoning is not preferred.

**Why this is the best next step:** it converts "this one model is faithful" into "faithfulness
varies systematically with post-training stage", using models that already exist. If the staircase
appears, it is a genuinely publishable result and directly actionable — it says *which* training
step costs you monitorability.

**If the staircase does not appear**, that is equally informative: it would mean our finding is
about scale or task, not pipeline.

**Changes needed:** `CHECKPOINT_REPO` per model; drop the checkpoint sweep to a single revision;
set `INSTRUMENTAL_PROMPT_STYLE = "chat_template"` for the ChatML models. Roughly a day's work,
mostly waiting.

**Cost:** 7 models × 76 scenarios × 5 passes = 2,660 judge calls ≈ **$13**, plus ~2–3 h GPU ≈ $4.

---

## Study B — Obfuscation pressure

**~$10 · ~1 h GPU · tests robustness, not just default behaviour**

[Zolkowski et al.](https://arxiv.org/abs/2510.19851) show monitoring works *without* obfuscation
pressure but can fail under it. Our study measured only the unpressured case.

Add a prompt condition instructing the model to conceal its reasoning, and measure:

| | expected if monitorable | expected if fragile |
|---|---|---|
| behavioural convergence | unchanged | unchanged |
| verbalization rate | unchanged | **drops sharply** |

**Question:** is this model's faithfulness a *property* or merely an *absence of effort*?

This is cheap, needs no training, and is the natural companion to Study A: A asks whether training
removes faithfulness, B asks whether prompting can.

**Cost:** 1 model × 76 × 5 × 2 conditions = 760 calls ≈ **$4**, plus ~1 h GPU.

---

## Study C — The Think trajectory

**~$40 · ~6 h GPU**

Repeat the full two-phase design on `Olmo-3-7B-Think` (55 checkpoints, subsample to ~15).

**Why:** our Phase 1 found no behavioural development — plausibly because maths RLVR simply does
not cultivate instrumental convergence. A *reasoning*-trained model with 3× the trajectory is the
fair test of the original hypothesis. It is also the model class where CoT monitoring actually
matters.

**This is the study that could still find the dissociation** the project originally set out to
detect. Our null may say more about maths RLVR than about RLVR.

---

## Study D — More scenarios

**~$15 · addresses the single biggest weakness**

Every claim rests on **6–7 convergent traces per category**. Nothing else in this list fixes that.

Options, in order of defensibility:
1. Add the three unused InstrumentalEval categories (34 more scenarios, already collected and
   judged behaviourally — only CoT judging is missing).
2. Extend the benchmark with new scenarios in the same style, validated against the originals.
3. Increase the checkpoint bracket from 5 to all 19 for Phase 2 — more observations, though *not*
   more independent scenarios, so it helps the bootstrap but not the clustering problem.

**Option 1 is nearly free** — the Phase 1 behavioural scores already exist for all 76 scenarios.
Only Phase 2 CoT judging is missing for the other 34.

---

## Study E — CoT monitor precision, as its own result

**~$5 · a practical safety finding hiding in the existing data**

We found **18 traces that verbalized shutdown-evasion reasoning and then complied** ([§4.2](02-results.md)).
A naïve monitor flagging "reasoning about evading shutdown" would fire 18 times for 7 real events —
a **72% false-positive rate**.

This is a deployable result: *verbalized consideration is not intent*, and CoT monitors need a
precision story, not just a recall story. It requires no new generation — only re-judging the
existing traces with a stricter "intent vs consideration" rubric.

Pairs naturally with [Emmons et al.](https://arxiv.org/abs/2510.23966) on monitorability metrics.

---

## Study F — The causal experiment

**~$200–2,000 · out of current budget, but this is the paper**

Fine-tune `Olmo-3-7B-RL-Zero-Math` with a CoT monitor folded into the reward, exactly as
[Baker et al.](https://arxiv.org/abs/2503.11926) describe, and re-measure faithfulness.

**Prediction:** verbalization collapses while behavioural convergence persists — obfuscated
convergence.

This converts every correlational claim in this project into a causal one, on a model where we have
a clean pre-intervention baseline. It requires RL training infrastructure rather than inference, so
it is an order of magnitude more expensive.

Studies A and B are the cheap observational shadows of this experiment. **Do them first** — if the
staircase in A appears, F becomes much easier to justify to a funder.

---

## Study G — Fix the capability control

**~$10 · methodological hygiene**

Re-run MATH at 4,096 tokens and report completion-conditional accuracy.

Our finding that a fixed-token-budget benchmark becomes a *length-endurance test* as RL lengthens
reasoning ([§6.1](02-results.md)) is a general warning for RLVR evaluation, and worth stating
cleanly with a properly measured curve behind it.

---

## Suggested sequence

| Order | Study | Cost | Establishes |
|---|---|---|---|
| 1 | **D-1** (3 unused categories) | ~$5 | Bigger n, using data you already have |
| 2 | **E** (monitor precision) | ~$5 | A standalone practical result, no new generation |
| 3 | **A** (post-training staircase) | ~$20 | *Which* training stage costs monitorability |
| 4 | **B** (obfuscation pressure) | ~$10 | Whether faithfulness is a property or an absence of effort |
| 5 | **C** (Think trajectory) | ~$40 | A fair test of the original hypothesis |
| 6 | **F** (causal) | $200+ | Causation — the actual paper |

**Studies 1–5 total roughly $80** and would take the work from "one model is faithful" to "here is
where in the post-training pipeline faithfulness is lost, whether prompting can break it, and what
it costs a monitor in false positives."

Against your remaining grant, that is affordable — and each step is independently publishable if the
next one stalls.

---

## What we would not do

- **More checkpoints on RL-Zero-Math.** The series is flat and the scenarios are exhausted; a
  twentieth checkpoint adds roughly one convergent trace.
- **A larger judge model.** Inter-judge κ = 0.78 is not the bottleneck; sample size is.
- **Scaling to 32B first.** `Olmo-3-32B-Think` exists, but scale is a confound to add *after* the
  pipeline question is settled at 7B, not before.
