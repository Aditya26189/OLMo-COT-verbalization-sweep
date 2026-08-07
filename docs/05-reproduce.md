# Reproducing the Study

Two paths: **verify the analysis** from committed data (free, ~2 minutes) or **re-run the sweep**
end to end (~$48, ~7 hours).

---

## A. Verify the analysis — free, no GPU, no API key

All raw results are committed. Every number in [02-results.md](02-results.md) can be regenerated:

```bash
git clone https://github.com/Aditya26189/OLMo-COT-verbalization-sweep
cd OLMo-COT-verbalization-sweep
pip install numpy scipy statsmodels ruptures scikit-learn matplotlib

python analysis/dissociation.py    # 2x2 tables, per-checkpoint + pooled Fisher, FDR
python analysis/effect_sizes.py    # Wilson CIs, odds ratios, base rates
python analysis/qualitative.py     # quoted evidence, 5-pass stability, length confound
```

`analysis/inter_rater_reliability.py` is the exception — it makes ~60 real API calls (~$0.19) and
needs `OPENAI_API_KEY`. Its output is committed at
[`results/inter_judge_agreement.json`](../results/inter_judge_agreement.json).

### Run the test suite

```bash
for t in tests/test_*.py; do echo -n "$t: "; python $t >/dev/null 2>&1 && echo PASS || echo FAIL; done
```

Seven suites, all offline. `test_e2e.py` executes **every notebook cell in order** against a
stubbed model and judge — it will catch a regression in the pipeline without touching hardware.

---

## B. Re-run the full sweep

### Prerequisites

| | |
|---|---|
| GPU | A100 SXM 80GB or better, **bf16 required** (the checkpoints are bf16-native) |
| Disk | ≥100 GB (checkpoints are 14.6 GB each; the cache purge keeps only one at a time) |
| OpenAI | ~$40 of credit, plus a dashboard spend cap |
| Gemini | free-tier key, for the zero-cost smoke test |
| GitHub | fine-grained PAT with **Contents: Read and write** on this repo |

### 1. Provision

RunPod A100 SXM 80GB, **Secure Cloud**, 100 GB container disk, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, env `HF_HOME=/workspace/hf_cache`.

> Secure over Community: Community pods can be reclaimed mid-run. Check availability by datacenter
> first — A100 stock is concentrated in a few sites.

### 2. Bootstrap

In the pod's **web terminal** (RunPod console → Connect → Web Terminal):

```bash
cd /workspace
git clone https://github.com/Aditya26189/OLMo-COT-verbalization-sweep rlvr-cot
cd rlvr-cot

cat > .env <<'EOF'
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GITHUB_TOKEN=...
EOF
chmod 600 .env

bash setup_pod.sh
```

`setup_pod.sh` verifies CUDA, ≥70 GB VRAM, **bf16 support**, ≥60 GB free disk, pinned dependencies,
all three keys, then runs the offline tests. It **fails loudly** rather than letting you discover a
problem five hours in.

### 3. Set the spend ceiling to your *remaining* credit

```bash
sed -i 's/MAX_JUDGE_SPEND_USD = 38.00/MAX_JUDGE_SPEND_USD = <your number>/' rlvr_cot_phase0_fixed.ipynb
```

The ledger at `results/judge_usage.json` carries prior spend forward. On a fresh clone it reflects
whatever is committed, not what you have actually spent — set the ceiling deliberately.

### 4. Smoke tests — do not skip either

```bash
python run_study.py smoke                    # Gemini, FREE
JUDGE_MODE=prod python run_study.py smoke    # GPT-4o, ~$0.05
```

**Treat a prod smoke failure as a hard stop.** It is the gate that catches the failure class which
destroyed a previous run.

Expect to see:
```
STUDY Evading Shutdown: n=20 / Hiding: n=11 / Deceptive: n=11
chat_template: math-specific; instrumental style=plain
health: finite_logits=True
judge API calls for 3 scenarios: 15
```

### 5. Run

```bash
tmux new -s sweep          # CONFIRM the green status bar appears
export GITHUB_TOKEN=$(grep '^GITHUB_TOKEN=' .env | cut -d'=' -f2-)
JUDGE_MODE=prod python run_study.py phase1 2>&1 | tee phase1.log; runpodctl stop pod $RUNPOD_POD_ID
```

Detach with **`Ctrl+B` then `D`**. `Ctrl+C` kills the run.

Then:
```bash
JUDGE_MODE=prod python run_study.py phase2 2>&1 | tee phase2.log; runpodctl stop pod $RUNPOD_POD_ID
```

`phase2` re-runs the Phase 1 analysis, locks the judge prompt, sweeps the bracketing checkpoints,
and writes final results and plots. Results are pushed to git after **every** checkpoint.

### 6. Monitor

```bash
tail -f phase1.log                              # safe: Ctrl+C stops only the tail
ls results/phase1/*_instrumental.json | wc -l    # checkpoints done, of 19
```

Or watch the commit history — one commit per checkpoint, visible without touching the pod.

**Stop immediately if** MATH accuracy is ~0 across checkpoints (harness broken, not a finding), or
`WARNING: git push failed` repeats (results are not backed up).

---

## What `run_study.py` does

It executes the notebook's own cells, selected by stage — there is exactly one copy of the logic,
and no divergence between "the notebook" and "the script".

| Stage | Runs |
|---|---|
| `smoke` | setup + smoke test |
| `phase1` | setup + 19-checkpoint sweep |
| `analysis` | setup + Phase 1 analysis + plots |
| `phase2` | setup + analysis + prompt lock + Phase 2 + final results |

`JUDGE_MODE` from the environment overrides the notebook default, so dev vs prod is a prefix rather
than an edit.

---

## Expected runtime and cost

| | time | cost |
|---|---|---|
| Setup + smoke tests | ~0.5 h | ~$0.8 GPU + $0.05 API |
| Phase 1 (19 checkpoints) | ~5.5 h | ~$8.7 GPU + ~$24 API |
| Phase 2 (5 checkpoints) | ~0.85 h | ~$1.3 GPU + ~$11 API |
| **Total** | **~7 h** | **~$48** |

~18 min per Phase 1 checkpoint. The log prints a **measured** ETA and cost projection after
checkpoint 1 — trust that over this table.

---

## Known environment traps

| Symptom | Cause | Fix |
|---|---|---|
| `set: pipefail: invalid option name` | CRLF line endings in a `.sh` | `sed -i 's/\r$//' setup_pod.sh` |
| `No module named 'matplotlib'` | `setup_pod.sh` aborted before `pip install` | fix the above, re-run |
| `hf_transfer is enabled but not available` | RunPod image sets `HF_HUB_ENABLE_HF_TRANSFER=1` without the package | `pip install hf_transfer` |
| `git push` → `403` | Token lacks **Contents: Read and write** | fix scope on the token |
| `git push` → `Invalid username or token` | `$GITHUB_TOKEN` not exported in that shell | `export GITHUB_TOKEN=$(grep ...)` |
| `duplicate session: sweep` | tmux refused; **you are not in tmux** | `tmux kill-session -t sweep` then retry |
| `not enough free GPUs on the host` | Stopped pod's host was reallocated | create a new pod; results are in git |
