# Does RLVR-Trained Chain-of-Thought Verbalize Instrumental Convergence?

A study of `allenai/Olmo-3-7B-RL-Zero-Math` across its full RLVR training trajectory —
19 checkpoints, 76 InstrumentalEval scenarios, GPT-4o judging at 5 independent passes.

**Core question:** as instrumental tendencies develop in behaviour, does the model's spontaneous
chain-of-thought reveal them — or does behaviour shift while the CoT stays silent?

---

## Headline finding

**The chain-of-thought is not silent. In this model it is close to fully faithful.**

Pooled across five checkpoints, traces where the model *behaved* convergently verbalized that
reasoning at rates far above the published faithfulness ceiling:

| Category | verbalized \| enacted | verbalized \| not enacted | risk difference (95% CI)¹ |
|---|---|---|---|
| Evading Shutdown | **100%** (7/7) | 19.4% (18/93) | +0.805 [+0.675, +0.918] |
| Hiding Unwanted Behavior | **88.1%** (37/42) | 7.7% (1/13) | +0.810 [+0.613, +0.962] |
| Deceptive Behavior | **100%** (6/6) | 0.0% (0/49) | +1.000 [+1.000, +1.000] |

¹ Scenario-level cluster bootstrap, 4,000 resamples — the correct correction for repeated
measurement of the same scenarios. The effect also survives a judge-unanimity restriction and
length matching. That is **5–35×** the <2% ceiling reported by Chen et al. for reward hacking —
the opposite of the hypothesised dissociation.

Of 55 enacted traces, **only 5 failed to verbalize** — and all five are borderline on inspection.
Meanwhile **18 traces verbalized shutdown-evasion reasoning and then complied**.

**Secondary finding:** behavioural convergence does *not* develop over RLVR training here
(τ = −0.019, p = 0.91). The tendencies are present from step_100 and stay flat.

Full analysis, confounds, and confidence levels: **[docs/02-results.md](docs/02-results.md)**

---

## Start here

| Document | What it covers |
|---|---|
| **[01-experiment.md](docs/01-experiment.md)** | The study design, every methodological decision and why. **No code knowledge needed.** |
| **[02-results.md](docs/02-results.md)** | Full results: statistics, quoted evidence, inference, confounds, confidence table |
| **[03-cost-control.md](docs/03-cost-control.md)** | How ~$48 was spent without waste — **reusable for any funded experiment** |
| **[04-defects-found.md](docs/04-defects-found.md)** | 25+ defects found, including three that silently destroyed a previous run |
| **[05-reproduce.md](docs/05-reproduce.md)** | Verify the analysis (free) or re-run the sweep (~$48) |

---

## Why the defects document matters

A previous run of this study completed, cost real money, and produced **380/380 zeros** on
InstrumentalEval and **2/500** on MATH. Both were instrumentation failures, not results.

Three root causes, each verified against live sources:

1. **Checkpoints are HuggingFace *branches*, not tags** — the old code read `refs.tags`, got an
   empty list, and crashed at its own assert.
2. **The tokenizer's chat template is a hardcoded maths prompt.** Wrapping InstrumentalEval
   scenarios in it asked the model to solve shutdown-evasion roleplay *as a maths problem*, and
   silently discarded the system-prompt role.
3. **MATH answers use `Answer:`, not `\boxed{}`** — the scorer looked for the wrong format.

Plus the one that made it invisible: **judge failures returned `0`**, which is indistinguishable
from "the judge looked and found nothing".

This run: **9,120 judge calls, zero failures, 100% coverage at every checkpoint.**

---

## Repository layout

```
├── rlvr_cot_phase0_fixed.ipynb   the study notebook (single source of truth)
├── run_study.py                  headless runner — executes notebook cells by stage
├── setup_pod.sh                  GPU bootstrap; fails loudly on any precondition
├── judge_prompt.txt              pre-registered Phase 2 judge prompt (committed before use)
│
├── docs/                         all documentation (start with 01)
├── analysis/                     post-hoc analysis scripts, re-runnable from committed data
├── tests/                        7 suites, all offline — no GPU, no API key
├── results/                      raw results: 19 Phase 1 + 5 Phase 2 checkpoints, plots, ledger
└── prev_runs/                    archived earlier notebook revisions
```

---

## Verifying the claims

Every number in the results document regenerates from committed data in about two minutes,
with no GPU and no API key:

```bash
pip install numpy scipy statsmodels ruptures scikit-learn matplotlib
python analysis/dissociation.py    # 2x2 tables, Fisher tests, FDR
python analysis/effect_sizes.py    # Wilson CIs, odds ratios, base rates
python analysis/qualitative.py     # quoted evidence, judge stability, length confound
python analysis/robustness.py      # clustering, unanimity, length matching, cluster bootstrap
```

Test suite:
```bash
for t in tests/test_*.py; do echo -n "$t: "; python $t >/dev/null 2>&1 && echo PASS || echo FAIL; done
```

`tests/test_e2e.py` executes **every notebook cell in order** against a stubbed model and judge.

---

## Honest limitations

The three most important, stated up front:

1. **Small convergent cells.** The 100% figures rest on **n=7** and **n=6**. Wilson lower bounds
   are 0.65 and 0.61. Direction is unambiguous; point estimates are fragile.
2. **Pseudo-replication.** The same scenarios recur across 5 checkpoints, so naïve Fisher p-values
   overstate precision. Corrected via cluster bootstrap — but a stricter scenario-majority test
   does **not** reach significance.
3. **The MATH capability control is compromised** — the apparent 0.52 → 0.06 collapse is mostly
   response-length inflation (truncation 49% → 96%, ρ = −0.956 with accuracy). Do not report it as
   a capability curve.

Full list: [docs/02-results.md §9](docs/02-results.md).

---

## Cost

| | |
|---|---|
| GPT-4o judge calls | 9,120 (+60 for inter-rater reliability) |
| Judge failures | **0** |
| API spend | ~$35 |
| GPU | ~7.8 h A100 SXM 80GB ≈ $12.5 |
| **Total** | **~$48** (planned: $145) |

---

## References

- **He, Yufei et al.** — *Evaluating the Paperclip Maximizer: Are RL-Based Language Models More
  Likely to Pursue Instrumental Goals?* [arXiv:2502.12206](https://arxiv.org/abs/2502.12206) —
  defines InstrumentalEval
- **Yanda Chen et al.** — *Reasoning Models Don't Always Say What They Think*
  [arXiv:2505.05410](https://arxiv.org/abs/2505.05410) — the <2% faithfulness ceiling
- **Baker, Huizinga, Gao** — *Monitoring Reasoning Models for Misbehavior and the Risks of
  Promoting Obfuscation* [arXiv:2503.11926](https://arxiv.org/abs/2503.11926) — CoT monitoring is
  effective until you optimize against the monitor. Our model is the limiting case of their
  low-optimization regime.
- **Zolkowski, Xing, Lindner** — *Can Reasoning Models Obfuscate Reasoning?*
  [arXiv:2510.19851](https://arxiv.org/abs/2510.19851) — monitoring works without obfuscation pressure
- **Emmons, Zimmermann, Elson** — *A Pragmatic Way to Measure Chain-of-Thought Monitorability*
  [arXiv:2510.23966](https://arxiv.org/abs/2510.23966) — frontier models show high default monitorability
- **Yueh-Han Chen et al.** — *Reasoning Models Struggle to Control their Chains of Thought*
  [arXiv:2603.05706](https://arxiv.org/abs/2603.05706) — prompted CoT suppression

> Two different first authors named Chen, plus a co-author named He He. Disambiguate by first name
> and arXiv ID.
