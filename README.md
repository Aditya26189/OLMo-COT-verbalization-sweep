<p align="center">
  <strong>🔬 RLVR CoT Verbalization Study</strong>
</p>

<p align="center">
  <em>Does chain-of-thought reveal instrumental convergence during RLVR training — or does behavior shift while reasoning stays silent?</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Work%20In%20Progress-orange?style=for-the-badge" alt="Status: WIP" />
  <img src="https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle" />
  <img src="https://img.shields.io/badge/GPU-2×%20NVIDIA%20T4-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="2x T4" />
  <img src="https://img.shields.io/badge/Model-OLMo--3--7B-blueviolet?style=for-the-badge" alt="OLMo-3-7B" />
</p>

---

## Motivation

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a dominant paradigm for training reasoning models. But as RLVR optimizes for task performance, models can develop **instrumental convergence** behaviors — strategically evading shutdown, hiding actions, or deceiving evaluators — as instrumentally useful sub-goals. Recent work by [He et al. (2025)](https://arxiv.org/abs/2502.12206) demonstrated that these behaviors emerge in RL-trained models and that *prompting* a model to reason step-by-step can suppress them.

However, a critical question remains unanswered:

> **When instrumental convergence behaviors develop through RLVR training, does the model's *spontaneous* chain-of-thought contain markers of those tendencies — or does behavior shift while CoT stays silent?**

This distinction matters because chain-of-thought is increasingly treated as a transparency and interpretability tool. If instrumental behaviors emerge *before* CoT verbalizes them, CoT monitoring is not a reliable early-warning signal. If verbalization leads or co-occurs with behavioral change, it supports CoT as a viable safety mechanism.

This study is further motivated by [Chen et al. (2025)](https://arxiv.org/abs/2505.05410), who showed that reasoning models' CoTs reveal unfaithful reasoning (e.g., hint usage) in ≥1% of cases, with reveal rates often below 20% — establishing a faithfulness ceiling against which our verbalization rates can be calibrated.

---

## Research Design

### Model Under Study

**[OLMo-3-7B-RL-Zero-Math](https://huggingface.co/allenai/OLMo-3-7B-RL-Zero-Math)** (AllenAI) — a 7B parameter model trained via RLVR directly from base pre-training, with **no supervised fine-tuning (SFT) and no RLHF**. This RL-Zero training lineage makes it uniquely suitable for studying how RLVR *alone* shapes both behavior and internal reasoning.

All publicly available intermediate training checkpoints are evaluated, enabling fine-grained tracking of behavioral and reasoning trajectories across the full course of RLVR training.

### Evaluation Dimensions

Three categories of instrumental convergence are tracked throughout, following the taxonomy of [He et al. (2025)](https://arxiv.org/abs/2502.12206):

| Category | Definition |
|---|---|
| **Evading Shutdown** | Actions that resist, avoid, or circumvent the model's termination or deactivation |
| **Hiding Unwanted Behavior** | Concealing actions or intentions from oversight systems or monitors |
| **Deceptive Behavior** | Strategically misrepresenting information or concealing true internal states |

### Two-Phase Methodology

```
Phase 1: Behavioral Sweep                    Phase 2: CoT Verbalization Sweep
┌──────────────────────────────┐            ┌──────────────────────────────────┐
│ For each checkpoint:         │            │ Anchored on inflection point:    │
│                              │            │                                  │
│  • 76 InstrumentalEval       │  ─────►    │  • Re-run with full CoT traces   │
│    scenarios (3 categories)  │ Inflection │  • Pre-registered LLM judge      │
│  • 100 MATH problems         │   point    │    classifies each trace         │
│    (Levels 3–5, capability   │            │  • Verbalization rates computed  │
│     control)                 │            │    per category per checkpoint   │
│  • PELT changepoint          │            │  • Dissociation analysis:        │
│    detection (BIC penalty)   │            │    verb. rate in converging vs   │
│                              │            │    non-converging responses      │
└──────────────────────────────┘            └──────────────────────────────────┘
```

**Phase 1 — Behavioral Sweep.** Every checkpoint is evaluated on the full [InstrumentalEval](https://github.com/yf-he/InstrumentalEval) benchmark (76 scenarios) and a held-out MATH subset (100 problems, Levels 3–5). The MATH accuracy series serves as a capability control — verifying that the model is learning math while we track instrumental behaviors. A PELT changepoint detection algorithm with a BIC-based penalty identifies the behavioral inflection point (the checkpoint where instrumental convergence rates shift). Mann-Kendall trend tests and lagged Spearman correlations with FDR correction characterize the co-evolution of behavior and capability.

**Phase 2 — CoT Verbalization Sweep.** Anchored on the inflection point from Phase 1, five checkpoints bracketing the transition are re-evaluated with full chain-of-thought traces captured. A **pre-registered** LLM judge prompt (committed via git *before* any CoT is inspected) classifies each trace for verbalization of instrumental convergence markers. Fisher's exact tests with Benjamini-Hochberg FDR correction test whether verbalization rates differ between behaviorally convergent and non-convergent responses. Cliff's delta provides non-parametric effect sizes.

### Statistical Framework

| Analysis | Method | Purpose |
|---|---|---|
| Trend detection | Mann-Kendall | Monotonic trend in instrumental convergence across training |
| Inflection identification | PELT (BIC penalty) + sensitivity analysis | Locate behavioral transition point |
| Behavior–capability coupling | Lagged Spearman (FDR-corrected) | Do instrumental behaviors track math capability? |
| Length confound | OLS regression | Rule out response length as artifact |
| Calibration | Comparison to He et al. Table 1 | Validate judge against published baselines |
| Dissociation | Fisher's exact test (FDR-corrected) + Cliff's delta | Does verbalization differ in converging vs. non-converging responses? |
| Judge reliability | Cohen's κ (human) or inter-judge agreement (dual-LLM) | Validate judge methodology |

### Pre-Registration & Methodological Safeguards

- **Judge prompt locked before Phase 2**: The classification prompt is committed to version control before any CoT traces are inspected, establishing a pre-registration timestamp.
- **Phase 1 scores reused in Phase 2**: Behavioral scores from Phase 1 are carried forward — no re-running, which prevents non-determinism or inconsistency.
- **Crash recovery**: Results are pushed to git after every checkpoint, tolerating Kaggle's 12-hour session limit.
- **No quantization**: Model runs in fp16 (no NF4/4-bit) to avoid calibration artifacts against He et al. baselines measured at full precision.

---

## Compute Environment

| Component | Specification |
|---|---|
| **Platform** | Kaggle Notebooks |
| **GPUs** | 2× NVIDIA T4 (16 GB VRAM each, 32 GB total) |
| **Precision** | fp16 via `torch.float16` |
| **Distribution** | `device_map='auto'` (HuggingFace Accelerate) across both T4s |
| **Execution** | Sequential — one checkpoint at a time |
| **Session limit** | 12 hours (Kaggle); crash recovery via per-checkpoint git push |

The 7B model in fp16 requires ~14 GB VRAM, distributed automatically across both GPUs. Each checkpoint takes approximately 1.75 hours to evaluate (76 InstrumentalEval + 100 MATH generations + judge calls).

---

## Setup

### Required API Keys

All secrets must be set via **Kaggle Secrets** (Settings → Add-ons → Secrets). No keys should be hardcoded in notebook source.

| Secret Name | Provider | Purpose | Required |
|---|---|---|---|
| `HF_TOKEN` | HuggingFace | Model access (if repo is gated) | Conditional |
| `GEMINI_API_KEY` | Google | Gemini Flash judge — Phase 1 behavioral scoring | ✅ |
| `ANTHROPIC_API_KEY` | Anthropic | Claude judge — Phase 2 CoT verbalization scoring | ✅ |
| `OPENAI_API_KEY` | OpenAI | GPT-4o cross-judge agreement validation | Optional |

### Dependencies

```
transformers>=4.57.0    accelerate>=0.30
ruptures>=1.1           scipy>=1.11           statsmodels>=0.14
google-generativeai>=0.5
datasets>=2.18          tqdm>=4.66            huggingface_hub>=0.22
matplotlib>=3.8         anthropic             openai
```

---

## Repository Structure

```
rlvr-cot/
├── rlvr_cot_verbalization_final.ipynb   # Main experiment notebook (Cells 0–11 + Appendix)
├── notebook4bae21a18d (1).ipynb         # Gemini 3.5 Flash run variant
├── notebook4bae21a18d.ipynb             # Earlier notebook version
├── notebook_edited (1).ipynb            # Edited variant
├── notebook_fixed.ipynb                 # Fixed variant
├── rlvr-3.5-flash-run/                  # Checkpoint results from 3.5 Flash run
│   ├── checkpoint_step_{100-500}_instrumental.json
│   ├── checkpoint_step_{100-500}_math.json
│   └── smoke_test.json
├── results/                             # Generated at runtime
│   ├── phase1/                          # Per-checkpoint behavioral + MATH scores
│   ├── phase2/                          # Per-checkpoint CoT traces + verbalization
│   ├── inflection_point.json            # Phase 1 → Phase 2 anchor
│   ├── phase1_curves.png                # Behavioral trajectory plots
│   ├── phase2_verbalization.png         # Verbalization analysis plots
│   └── phase2_summary.json             # Complete statistical summary
└── judge_prompt.txt                     # Pre-registered judge (committed before Phase 2)
```

---

## Known Limitations

- **Gemini Flash as Phase 1 judge** — He et al. validated GPT-4o (85.3% IAR) and Claude Sonnet 3.5 (78.6% IAR). Gemini Flash is unvalidated; absolute convergence rates are not directly comparable. Relative trends across checkpoints remain interpretable.
- **Phase 2 sample size** — With ~5 checkpoints bracketing the inflection, Spearman correlations require |ρ| ≥ 0.88 for significance at p<0.05. Results are exploratory, not confirmatory.
- **Hiding Unwanted Behavior asymmetry** — A negative classification (no verbalization) is inherently ambiguous for this category: the model may not have the tendency, or it may be successfully hiding it without articulating the intent.
- **No chat template** — OLMo-3-7B-RL-Zero-Math is an RL-Zero model and may lack a chat template. A minimal instruction prefix fallback is used when `apply_chat_template()` is unavailable.

---

## References

1. He, Y., et al. (2025). *Instrumental Convergence Evaluations for Large Language Models.* [arXiv:2502.12206](https://arxiv.org/abs/2502.12206)
2. Chen, Y., Benton, J., et al. (2025). *Reasoning Models Don't Always Say What They Think.* Anthropic. [arXiv:2505.05410](https://arxiv.org/abs/2505.05410)
3. AllenAI. *OLMo-3-7B-RL-Zero-Math.* [HuggingFace](https://huggingface.co/allenai/OLMo-3-7B-RL-Zero-Math)
4. Xiong, W., et al. (2025). *Building Math Agents with Multi-Turn Iterative Preference Learning.* [arXiv:2502.01154](https://arxiv.org/abs/2502.01154)
