# Cost Control — RunPod GPU and OpenAI API

*Written to be reusable. If you run another funded experiment, read this first — the failures
below are generic, not specific to this study.*

**Outcome:** the full study cost **~$48** (~$35 OpenAI, ~$12.5 GPU) against a plan of ~$145,
with **zero wasted API calls** and one interrupted run that lost nothing.

---

## 1. The failure this was all designed to prevent

A previous run of this study completed successfully, cost real money, and produced:

- **380 / 380** InstrumentalEval scores == `0`
- **2 / 500** MATH correct (0.4%)

Neither was a result. Both were instrumentation failures ([04-defects-found.md](04-defects-found.md)).
The run *looked* fine while producing nothing.

**This is the failure mode that matters.** A crash is cheap — you notice it. A run that completes
and returns plausible-looking garbage costs the full budget *and* the time spent believing it.

Every guard below exists to make that specific outcome impossible.

---

## 2. What cost money, and what actually happened

| | planned | actual |
|---|---|---|
| GPT-4o judge | $124 | **~$35** |
| GPU | $21 | **~$12.5** |
| **Total** | **$145** | **~$48** |

The plan overestimated API by **3.6×**. It assumed ~$0.0136/judge-call; the measured rate was
**$0.0033** (Phase 1) and **$0.0058** (Phase 2). Phase 1 calls are dominated by the model response
in the prompt, and the judge's own output is tiny (~26 tokens).

**Lesson:** estimate cost from *measured token counts on a real call*, not from a per-call guess.
One prod smoke test would have corrected a 3.6× error before committing.

### Where GPU time actually went

| | hours | cost |
|---|---|---|
| Phase 1 sweep (19 checkpoints) | ~5.5 | $8.75 |
| Setup, failed attempts, idle | **~2.3** | **$3.66** |
| Phase 2 (5 checkpoints, fresh pod) | ~0.85 | $1.33 |

**~30% of GPU spend was idle time during setup fumbling**, not computation. On a small budget that
is the largest single line item after the sweep itself. See §6.

---

## 3. The guards, and which ones fired

| Guard | What it prevents | Fired? |
|---|---|---|
| `MAX_JUDGE_SPEND_USD` | Hard dollar ceiling from **real** API token counts | No — but bounded everything |
| Persisted ledger (`results/judge_usage.json`) | The ceiling silently re-arming after a restart | **Yes — twice** |
| `JUDGE_CALL_BUDGET` | Runaway call volume, counting retries | No |
| Failure circuit breaker (>20%) | The 380-zeros failure mode | No (0 failures) |
| Per-checkpoint coverage floor (80%) | Saving a mostly-unjudged checkpoint | No |
| Judge health check before every sweep | Starting a 5-hour run on a dead key | **Yes — caught a retired model** |
| Independent instrumental/MATH skip | Re-paying ~380 judge calls for finished work | **Yes — saved ~$1.25** |
| Per-checkpoint git push | Losing hours of paid work to one crash | **Yes — twice** |
| Disk preflight + HF cache purge | Dying on a full disk mid-sweep | Yes (silently) |
| Measured cost projection each checkpoint | Learning the real rate at checkpoint 19 | Yes |
| Pod self-stop on exit | Idle GPU burning overnight | **Yes — twice** |

### 3.1 The dollar ceiling must use real token counts

A call-count cap is a proxy. Cost per call depends on how long the model's response was, which you
do not know in advance. The OpenAI response carries `usage.prompt_tokens` and
`usage.completion_tokens` — book those:

```python
u = result.usage
_record_tokens(u.prompt_tokens, u.completion_tokens)   # BEFORE returning
```

Book them *before* checking whether the content is usable. You are billed for a garbage response
too.

### 3.2 The ledger must survive process restart

An in-memory counter resets to zero every time the process starts. Three restarts = three times the
intended spend, silently. Persist it:

```json
{"attempts": 381, "calls": 381, "tokens_in": 449119,
 "tokens_out": 13169, "spend_usd": 1.2545}
```

**This fired twice** — the sweep was killed and restarted, and the ledger correctly carried prior
spend into the new budget both times.

**But we also got this wrong:** `judge_usage.json` was never added to the git push list, so only a
stale early copy reached the repo. After the pod was destroyed, ~$23 of accounting was lost. The
fix is one line — **push the ledger with the results.**

### 3.3 Failure must never be scored as a result

The single most important line in the codebase:

```python
except Exception:
    return None      # NOT 0
```

`0` means "the judge looked and found nothing". `None` means "we don't know". Conflating them is
what produced 380 false zeros. Downstream, `None` is excluded from rates and counted separately;
if too many accumulate, the run stops.

### 3.4 Guards must be tested, not assumed

A guard that doesn't fire is worse than none — it provides false confidence. Each was tested by
deliberately breaking things ([`tests/test_failures.py`](../tests/test_failures.py)):

| Scenario | Asserted |
|---|---|
| Judge dead from the start | Health check stops it in **≤5 calls** |
| Judge dies mid-sweep | Breaker fires; **no partly-unjudged checkpoint saved** |
| 401 auth error | Aborts in **<40 attempts** — no retry storm |
| Dollar ceiling reached | Spend never *exceeds* the ceiling |
| GPU budget exhausted | Stops **before loading a model** |
| Intermittent 429s | Retries absorb them; run completes |

Two of my own assertions were wrong by one call. Checking is what caught it — the guards were
right, my expectations weren't.

---

## 4. Efficiency, not just safety

Waste is not only overspending — it is also paying for work you didn't need to do.

| Change | Effect |
|---|---|
| **Batched generation** (left-padded, batch 16) | Several-fold GPU throughput vs one-at-a-time |
| **Pooled judge calls** | GPU idles during judging but still bills. Serial judging cost ~2.5h of paid idle; pooling at concurrency 16 cut it to ~0.8h (~$2.70) |
| **Prompt-cache staging** | The 5 passes send a byte-identical prompt. Firing pass 0 for all scenarios first, then 1–4, makes them eligible for 50%-off cached input — ~$10 on paper |
| **Independent phase skip** | A rerun after a partial failure re-judged nothing |
| **HF cache purge** | 19 × 14.6 GB = 277 GB would have filled the disk at checkpoint ~4 |

### The batch-size calculation worth reusing

`Olmo-3-7B` is **MHA, not GQA** (`num_key_value_heads == num_attention_heads == 32`), so KV cache
is `2 × 32 layers × 32 heads × 128 dim × 2 bytes` = **512 KB per token**. At a 2,500-token sequence
that is **1.25 GB per sequence**, on top of 14.6 GB of weights:

```
VRAM ≈ 14.6 + 2 + 1.25 × batch  GB
24 GB → batch 4 | 32 GB → 8 | 48 GB → 16 | 80 GB → 16 (capped)
```

Capped at 16 deliberately: HF `generate()` runs every sequence until the **slowest** finishes, so
a bigger batch buys throughput but wastes more on stragglers.

**Always check GQA vs MHA before sizing a batch.** Assuming GQA here would have overestimated
capacity by ~4×.

---

## 5. Choosing the GPU — the analysis that mattered

Modelled from memory bandwidth (decode is bandwidth-bound), with 4.95M tokens to generate:

| GPU | VRAM | batch | tok/s | hours | **$** | availability |
|---|---|---|---|---|---|---|
| A40 | 48 | 16 | 111 | 14.0 | 6.15 | MED |
| **A100 SXM** | **80** | **16** | **459** | **4.6** | **7.29** | **HIGH** |
| H100 SXM | 80 | 32 | 753 | 3.4 | 11.23 | HIGH |
| RTX 4090 | 24 | 4 | 59 | 25.1 | **18.54** | HIGH |
| L4 | 24 | 4 | 17 | 80.4 | 39.42 | MED |

**The headline: GPU cost was never the binding constraint.** Every viable option landed between
$6 and $20 — against ~$35 of API spend. Do not over-optimise this.

**The RTX 4090 trap:** the obvious "cheap" pick is second-worst overall. 24 GB forces batch 4 and
destroys throughput. **Cheap per hour, expensive per token.** Always compare cost-per-token, not
cost-per-hour.

Why A100 SXM won: HIGH availability (a cheaper GPU you cannot provision is worth nothing), 80 GB
removing all OOM risk, and ~4.6 h fitting one supervised session. A40 saved $1.14 but took 14
hours — and one forgotten idle day would erase that saving nine times over.

---

## 6. Operational mistakes — where the money really went

The code guards worked. **The losses were operational.** In order of cost:

### 6.1 Idle pod during setup — ~$3.66

The pod bills from boot. Between CRLF errors, dependency installs, git auth, and a `Ctrl+C`
mistake, ~2.3 hours were billed for nothing.

**Prevention:** get the pod to a working state with a *scripted, tested* bootstrap
([`setup_pod.sh`](../setup_pod.sh)) rather than debugging interactively at $1.59/hr. Better still,
validate everything you can locally first — all 7 test suites run with no GPU and no API key.

### 6.2 `Ctrl+C` killed a running sweep — ~1 hour

A long run was started **outside tmux** (`tmux new` silently refused: `duplicate session`), then
killed when the terminal was reused for git commands.

**Prevention:**
- `Ctrl+B` then `D` = detach (survives). `Ctrl+C` = kill.
- **Verify you are in tmux** — there is a green status bar. `duplicate session` means you are *not*.
- Need a shell while it runs? `Ctrl+B` then `c` for a new window; never reuse the run's window.

### 6.3 Stopping the pod lost the GPU — forced a rebuild

A stopped RunPod pod stays pinned to one host. Restarting requires *that machine* to have a free
GPU:

> `There are not enough free GPUs on the host machine to start this pod.`

The datacenter showed MEDIUM availability overall; the specific host was full.

**Prevention:** treat "stop" as "may not get it back". On scarce hardware, either keep it running
or accept a rebuild. Because results were pushed per checkpoint, the rebuild cost ~15 minutes
instead of the whole run.

### 6.4 Auto-stop is mandatory, not optional

```bash
python run_study.py phase1 2>&1 | tee phase1.log; runpodctl stop pod $RUNPOD_POD_ID
```

**`;` not `&&`** — it must stop on failure too, which is exactly when you are not watching. At
$1.59/hr a forgotten weekend is $114. This fired correctly twice, including once at 04:52 with
nobody watching.

---

## 7. Credential handling

Two defects, both blocking, both found before the run:

1. **`.env` held a bare secret with no `NAME=` prefix** — nothing would have read it.
2. **The notebook never read `.env` at all**, only `os.environ`. Either alone gives
   *"OPENAI_API_KEY not set"* on the pod.

Fixed with a `load_dotenv()` that prints **names only, never values**, and lets real environment
variables win.

Also: `.gitignore` covered `.env` and `*.env` but **not** `.env.bak` or `.env.local`. Added
`.env*` and `*.env.*`.

**A hard rule, learned the expensive way:** paste secrets into the pod terminal or a file — never
into a chat, an issue, or a commit. Keys pasted into a conversation must be treated as compromised
and rotated. To verify a key is correct without exposing it, check its *shape* (prefix and length)
and then make one live call.

### A model listed by the API is not necessarily callable

`gemini-2.5-flash-lite` appears in `models.list` but 404s on an actual call:
*"no longer available to new users."* Only a live `generateContent` proves anything. Pinned model
names go stale; a rolling alias (`gemini-flash-lite-latest`) does not.

---

## 8. Checklist for the next funded experiment

**Before spending anything**
- [ ] Every guard tested by deliberately breaking it
- [ ] Full pipeline dry-run with stubbed model and API (catches integration bugs unit tests miss)
- [ ] Dollar ceiling set from **measured** token counts, sized to *remaining* credit
- [ ] Ledger persisted **and pushed** with results
- [ ] Failures return `None`, never `0`
- [ ] Provider dashboard spend cap set — independent of anything your code tracks
- [ ] Credentials load correctly; no secrets in git; `.env*` ignored
- [ ] Scripted bootstrap that fails loudly, tested locally

**Before the long run**
- [ ] Free-tier smoke test (validates wiring at zero cost)
- [ ] Paid smoke test — **treat failure as a hard stop**
- [ ] Confirm you are in tmux (green bar)
- [ ] Self-stop appended with `;` not `&&`
- [ ] Per-checkpoint result push working (watch for a push failure on checkpoint 1)

**During**
- [ ] Check measured cost/ETA after checkpoint 1 — do not wait until the end
- [ ] Watch for the metric that means "harness broken", not just "results bad". Here: MATH ≈ 0

**After**
- [ ] Reconcile actual spend against the provider dashboard
- [ ] Stop or terminate the pod
- [ ] Verify all artifacts exist and are non-degenerate before trusting them

---

## 9. The one-sentence version

Guard against *plausible garbage*, not just crashes: make failure impossible to mistake for a
result, cap spend in dollars from real token counts, persist that accounting across restarts, push
results incrementally — and remember that on a short experiment, **idle time and operator error
cost more than the compute**.
