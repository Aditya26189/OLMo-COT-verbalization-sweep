# Results — RLVR CoT Verbalization Study

**Model:** `allenai/Olmo-3-7B-RL-Zero-Math` · 19 RLVR checkpoints (`step_100` … `step_1900`)
**Benchmark:** InstrumentalEval (He, Yufei et al., [arXiv:2502.12206](https://arxiv.org/abs/2502.12206)), 76 scenarios
**Judge:** GPT-4o, 5 independent passes per item, majority vote
**Run:** 2026-08-06/07 · Phase 1 19/19 checkpoints · Phase 2 5 checkpoints bracketing the anchor

---

## Headline

**The chain-of-thought is not silent. In this model it is almost perfectly faithful.**

Pooled across the five Phase 2 checkpoints, whenever the model *behaved* convergently, its
spontaneous CoT *said so* — at rates far above the published faithfulness ceiling.

| Category | verbalized \| behaviour=1 | verbalized \| behaviour=0 | Fisher p | odds ratio |
|---|---|---|---|---|
| Evading Shutdown | **100.0%** (7/7) | 19.4% (18/93) | 0.00003 | ∞ |
| Hiding Unwanted Behavior | **88.1%** (37/42) | 7.7% (1/13) | <0.00001 | 88.8 |
| Deceptive Behavior | **100.0%** (6/6) | 0.0% (0/49) | <0.00001 | ∞ |

All three survive Benjamini–Hochberg FDR correction.

This is the **opposite** of the dissociation the study was designed to detect. The pre-registered
question was *"does behaviour shift while CoT stays silent?"* For this model the answer is no —
behaviour and CoT move together.

---

## Phase 1 — behavioural convergence does not develop over RLVR training

```
InstrumentalEval mean:   0.276 – 0.368   (flat)
Mann-Kendall:            tau = -0.019,  p = 0.91
PELT changepoints:       none
anchor:                  step_1500 (max-gradient fallback, pre-registered)
judge coverage:          100% at every checkpoint (0 failures / 7,220 calls)
```

There is **no monotonic trend and no changepoint**. Instrumental convergence wobbles around
~0.32 from step_100 to step_1900. PELT found nothing at the BIC penalty; the pre-registered
max-gradient fallback fired and selected `step_1500`.

Per category across all 19 checkpoints:

| Category | n | mean | range | Kendall tau | p |
|---|---|---|---|---|---|
| Evading Shutdown | 20 | 0.055 | 0.00–0.15 | +0.22 | 0.24 |
| Hiding Unwanted Behavior | 11 | 0.780 | 0.64–0.91 | −0.34 | 0.063 |
| Deceptive Behavior | 11 | 0.110 | 0.09–0.18 | +0.08 | 0.69 |

**This null is what licenses pooling in Phase 2.** With no temporal trend, the five checkpoints
are effectively replicates rather than an ordered series, so pooling is a property of the data
rather than a convenience.

---

## Phase 2 — verbalization rates

| Checkpoint | Evading Shutdown | Hiding Unwanted | Deceptive |
|---|---|---|---|
| step_1300 | 0.200 | 0.909 | 0.182 |
| step_1400 | 0.300 | 0.545 | 0.091 |
| step_1500 | 0.150 | 0.636 | 0.091 |
| step_1600 | 0.300 | 0.636 | 0.091 |
| step_1700 | 0.300 | 0.727 | 0.091 |

### Versus Chen et al. (arXiv:2505.05410)

Chen et al. found CoTs verbalize reward hacks in **fewer than 2% of examples in 5 of 6 RL
environments**. This study, on spontaneous CoT under InstrumentalEval:

| Category | overall rate | multiple of the 2% ceiling |
|---|---|---|
| Evading Shutdown | 25.0% | **12×** |
| Hiding Unwanted Behavior | 69.1% | **35×** |
| Deceptive Behavior | 10.9% | **5×** |

The gap is large enough that it is probably a difference in *kind*, not degree. Two candidate
explanations, both testable and neither settled here:

1. **Task construction.** Chen et al. measured verbalization of a *reward hack* — behaviour the
   training signal rewards while the CoT has no incentive to mention. InstrumentalEval scenarios
   invite explicit strategic reasoning, so verbalization is closer to the task than a side effect.
2. **Model scale and training.** A 7B RL-Zero math model has no RLHF pressure to suppress
   self-incriminating reasoning. The frontier models Chen et al. studied do.

### Per-checkpoint tests are underpowered — pooled tests are not

Of the 14 per-checkpoint Fisher tests, **0 survive FDR** (smallest p_fdr = 0.064). Cell counts are
the reason: several checkpoints have only 1–2 behaviourally-convergent traces in a category.

Pooling to n=7/42/6 convergent traces gives p < 0.0001 across all three categories. **Report the
pooled result; report the per-checkpoint tests as descriptive only.**

---

## The MATH capability control is confounded — do not report it as-is

```
MATH accuracy:            0.520 -> 0.060     tau = -0.838,  p < 0.00001
MATH truncation:          49%   -> 96%
truncation vs accuracy:   rho = -0.956,      p < 0.00001
accuracy | finished only: 0.96  -> 0.75      tau = -0.534,  p = 0.0016
```

The apparent 8× capability collapse is **mostly an artifact of response-length inflation**. RLVR
lengthens the CoT; by step_1900, 96% of MATH generations never reach their `Answer:` line within
2,048 tokens and are scored 0. The correlation between truncation and measured accuracy is −0.956.

A real decline exists (0.96 → 0.75 among completed generations) but is far smaller than the
headline number.

**Do not publish 0.52 → 0.06 as a capability curve.** Either report accuracy conditional on
completion, or re-run MATH with a much larger token budget.

The instrumental series is *not* affected: response length grew only 12% (694 → 775 words) and
the length regression is null (coef ≈ 0.00002, p = 0.94).

---

## Calibration against He et al.

| | Hiding Unwanted Behavior |
|---|---|
| This study, final checkpoint | **0.727** |
| He et al., RL-model average | 0.5637 |
| He et al., RLHF arm | 0.3333 |
| delta vs RL baseline | **+0.164** |

Above the RL baseline, in the same direction as their RL-vs-RLHF contrast. The delta exceeds the
0.15 threshold and needs addressing in any write-up.

**A larger discrepancy:** Evading Shutdown here is **0.055**, where He et al.'s table shows RL
models at 35–50%. The most likely cause is that this is a *math*-RL model with no agentic
training, prompted in plain format rather than a chat template. State it as a limitation — the
absolute rates are not directly comparable to theirs, only the relative structure is.

---

## Limitations

1. **Small cell counts.** Only 7 of 100 Evading Shutdown traces and 6 of 55 Deceptive Behavior
   traces were behaviourally convergent. The 100% verbalization figures rest on n=7 and n=6.
2. **Pooling assumption.** Justified by the Phase 1 null, but it does assume checkpoints are
   exchangeable.
3. **Single judge model.** GPT-4o only. The inter-rater reliability cell (`compute_inter_judge_agreement`)
   has not been run — do this before publication.
4. **Prompt format deviation.** The model's chat template hardcodes a math-solving instruction, so
   scenarios were presented in plain format instead. Necessary, but a deviation from He et al.
5. **Behavioural score reuse.** Phase 2 reuses Phase 1 behavioural scores against freshly generated
   traces. Both are greedy decodes of the same prompt, so the Phase 2 trace is a longer superset of
   the Phase 1 response — but they are not byte-identical generations.
6. **No trend to explain.** The headline dissociation question presupposed a behavioural shift that
   did not occur in this model.

---

## Cost and infrastructure

| | |
|---|---|
| GPT-4o judge calls | 9,120 (7,220 Phase 1 + 1,900 Phase 2) |
| Judge failures | **0** |
| API spend | ~$35 of $43 |
| GPU | ~7.8 h A100 SXM 80GB ≈ $12.5 of $15 |
| **Total** | **~$47** |

Phase 2 alone: 2,282 calls, 3.82M input / 0.27M output tokens, $12.19 ledgered, **0 failures**.

---

## Next steps

1. **Run inter-rater reliability** — `compute_inter_judge_agreement()`. Single-judge results are
   not defensible for publication.
2. **Re-run MATH with a larger token budget**, or restate the control as completion-conditional.
3. **Consider a second model family.** The Phase 1 null may be specific to a math-RL model; a
   code or general RL-Zero variant would test whether the flat behavioural profile generalises.
4. **Sample traces by hand.** With verbalization this high, read a few — 100% verbalization on
   n=7 warrants eyeballing the actual text before it goes in a paper.
