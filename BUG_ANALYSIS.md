# `rlvr_cot_phase0_fixed.ipynb` — Bug Analysis

**Date:** 2026-08-05 · **Verdict: the notebook was NOT ready to run.** It crashed at Cell 3, and
had it not crashed there, it would have burned the full GPU + API budget producing zeros.

Original preserved at [prev_runs/rlvr_cot_phase0_fixed.ORIGINAL.ipynb](prev_runs/rlvr_cot_phase0_fixed.ORIGINAL.ipynb).
Rebuilt notebook: [rlvr_cot_phase0_fixed.ipynb](rlvr_cot_phase0_fixed.ipynb).

---

## The decisive evidence

`rlvr-3.5-flash-run/` holds the output of a previous real run:

| File | Result |
|---|---|
| 5 × `*_instrumental.json` | **380 / 380 scores == 0** |
| 5 × `*_math.json` | **2 / 500 correct (0.4%)** |

Neither is a finding. An RLVR *math* model does not score 0.4% on MATH, and a 76-scenario
adversarial benchmark does not return a perfect run of negatives. Both numbers are instrumentation
failures, and I traced each to a specific line. That run's GPU hours and API calls bought nothing.

Everything below was verified against live sources (HuggingFace API, the InstrumentalEval repo,
the checkpoint's own `chat_template.jinja`), not inferred.

---

## Critical — would have wasted the grant

### C1. Checkpoints are branches, not tags → immediate hard crash
`ALL_CHECKPOINTS = sorted(tags, ...)` then `assert len(ALL_CHECKPOINTS) > 0`.

The HF API for `allenai/Olmo-3-7B-RL-Zero-Math` returns **0 tags and 20 branches**
(`main`, `step_100` … `step_1900`). The old Cell 3 always produced `[]` and tripped its own
assert. **The notebook could not reach Cell 4.**

*Fixed:* `enumerate_checkpoints()` reads both refs, auto-selects whichever holds `step_*`
revisions, excludes `main`, sorts numerically. Yields the expected 19 checkpoints.

### C2. Category names never matched → every per-category metric silently zero
The benchmark ships `Alignment Drift - Evading Shutdown.csv`, so `csv_file.stem` is
`"Alignment Drift - Evading Shutdown"`. That was compared against `"Evading Shutdown"`:

```python
if item["category"] in CATEGORIES:   # never true
```

Confirmed in the prior run's JSON — every record carries the full stem. Consequences: all three
per-category series flat at 0, the He et al. calibration check comparing 0.000 against 0.5637, the
human-annotation sampler returning nothing, and **the entire Phase 2 verbalization analysis
producing zeros** — Phase 2's whole output is filtered by `r['category'] == cat`.

*Fixed:* `map_category()` strips the prefix; scenarios carry `raw_category` **and** `category`;
the loader asserts all three study categories resolved and prints per-category `n`
(**Evading Shutdown 20, Hiding Unwanted Behavior 11, Deceptive Behavior 11** — 42 of 76).

### C3. The chat template is a math prompt — applied to shutdown-evasion scenarios
`chat_template.jinja` on every `step_*` revision is, verbatim:

```
Solve the following math problem step by step. The last line of your response should be
the answer to the problem in form Answer: $Answer ...
{message contents concatenated — roles discarded}
Remember to put your answer on its own line after "Answer:"
```

`wrap_scenario()` called `apply_chat_template()` on InstrumentalEval scenarios. So every scenario
was prefixed *"Solve the following math problem"*, and because the template only concatenates
`message['content']`, **the He et al. system prompt's `system` role was silently discarded.** The
model was asked to solve a shutdown-evasion roleplay as a math problem. This alone explains the
380 zeros: the judge was correctly reporting no convergence in math-flavoured non-answers.

*Fixed:* `INSTRUMENTAL_PROMPT_STYLE="plain"` presents the system prompt + scenario directly.
The notebook auto-detects a math-specific template and **refuses** to use it for scenarios. The
smoke test prints both prompts so you can see what the model actually receives.

### C4. MATH scorer looked for `\boxed{}`; the model emits `Answer:`
Same template — it instructs `Answer: $ANSWER`. `score_math()` extracted only `\boxed{}`, fell
through to "last non-empty line", and matched almost nothing. **This is the 0.4%.**
Compounded by `max_new_tokens=512`, which truncates this long-CoT model before it reaches its
answer line at all.

*Fixed:* `extract_prediction()` reads `Answer:` first, then `\boxed{}`, then last line. Gold still
comes from `\boxed{}` (correct for the dataset). `MATH_MAX_NEW_TOKENS=1536`, truncation counted and
reported, and `check_math_failures()` shouts if accuracy is near zero on a math-RL model.

### C5. Judge failures were scored as "no convergence"
```python
except Exception as e:
    print(f'Judge error ...'); return 0     # indistinguishable from a real negative
```
Plus a strict `decision == 'convergence'` parse that fails on `DECISION: [convergence]` —
the exact bracketed form the prompt template asks for.

*Fixed:* failures return `None`, never `0`. `_parse_decision()` tolerates brackets, bold, spacing
and hyphens (and checks `no_convergence` before `convergence`, since one contains the other).
A **circuit breaker** aborts once >20% of calls fail, and each checkpoint refuses to save below 80%
judge coverage. A dead judge now stops the run instead of filling 14 GPU-hours with zeros.

### C6. The cross-model pivot ran on every single run
```python
if inflection_data.get("kill_condition", False):
    print("Kill condition triggered. Skip pivot execution.")
else:
    ... downloads 3 models, runs 76 scenarios × 5 passes each ...
```
Inverted. `kill_condition` is always `False` under the max-gradient fallback, so the `else` branch
— the pivot — **always executed**: three extra 14.6 GB downloads, hours of GPU, and ~1,140
unbudgeted judge calls, every time the notebook was run to the end.

*Fixed:* explicit `RUN_CROSS_MODEL_PIVOT = False` opt-in, with a budget assertion before it starts.

### C7. No spend ceiling anywhere
Nothing bounded total API calls. A retry loop, a re-executed cell, or a ThreadPoolExecutor bug
would bill without limit.

*Fixed:* `JUDGE_CALL_BUDGET = 12000` hard stop (planned load is 9,180), enforced per attempt
including retries, plus a printed dollar estimate before the sweep.

---

## High — silently wrong results

### H1. `PELT(jump=5)` — the ruptures default — could not find most changepoints
`rpt.Pelt(model='rbf')` uses `jump=5`, so on a 19-point series a changepoint can only be placed at
index 5, 10 or 15. I verified this on synthetic data with a true break at index 7:

```
jump=1 changepoints: [7, 19]   <- correct
jump=5 changepoints: [5, 19]   <- old default, off by two checkpoints
```

Two checkpoints off sends Phase 2 to the wrong bracket — i.e. the entire Phase 2 budget spent on
the wrong checkpoints. *Fixed:* `jump=1, min_size=2`.

### H2. `model='rbf'` with a BIC penalty derived for least squares
`pen = log(n)·σ²` is the BIC penalty for an **l2** cost. Pairing it with the rbf kernel cost mixes
units, so the penalty does not mean what the comment claims — a reviewer will catch this.
*Fixed:* `model='l2'`, matching the penalty. Zero-variance guard added.

### H3. Phase 2 indexed the wrong list
`inflection_idx` indexes `steps` (checkpoints that produced results), but Phase 2 did
`ALL_CHECKPOINTS[inflection_idx]`. The lists diverge the moment one checkpoint fails — and
`load_phase1_results()` had an explicit `continue` for that case. Silently wrong checkpoints.
*Fixed:* `analyzed_steps` is persisted in `inflection_point.json`; Phase 2 indexes that.

### H4. Phase 2 ↔ Phase 1 joined by list position
`behavioral_score = p1_scores[idx]["score"]` assumed identical ordering across two separate runs,
with no scenario identity stored to check it.
*Fixed:* stable `scenario_idx` written in Phase 1 and joined on in Phase 2, with an assertion that
refuses to run against result files from the older revision.

### H5. Cohen's kappa paired human labels against unrelated traces
`compute_cohen_kappa()` took the **first** verbalization record matching the category, searched the
trace file for a text-prefix match, then took the judge score from that arbitrary first record —
not from the matched trace. The resulting kappa was meaningless.
*Fixed:* stable `trace_id` in both files, joined directly.

### H6. Truncated judge output parsed as a failure → scored as all-zeros
`judge_trace()` used `max_tokens=1000` for a prompt requiring Step 1 quotations *and* a JSON block;
truncation lost the JSON. The `re.search(r'\{[\s\S]*\}')` fallback also breaks whenever a Step 1
quotation contains a brace.
*Fixed:* 1500 tokens, and `_parse_judge_json()` scans brace-balanced, string-aware candidates and
takes the last one that parses and carries a study category. Handles fenced blocks and trailing prose.

### H7. Lexicographic re-sort after subsampling
`sorted(set(early + mid + late))` drops the numeric key: `step_1000 < step_200 < step_2000`.
Scrambles the training-time axis the whole changepoint analysis rests on. Latent at N=19 (no
subsampling), fatal if the repo ever publishes more. *Fixed:* numeric key retained.

### H8. fp16 on a bf16 checkpoint, unchecked
`config.json` declares `dtype: bfloat16`. Forcing fp16 (T4 has no bf16) risks overflow to inf/NaN
and silent garbage for an entire sweep — a plausible additional contributor to the previous zeros.
*Fixed:* bf16 auto-selected where supported; on fp16 hardware `preflight_model_health()` asserts
finite logits and non-empty generation before any real work.

### H0. The HF cache would fill the disk and kill the sweep at checkpoint ~4
Each checkpoint is **14.6 GB** of weights and the HuggingFace cache never evicts on its own.
19 checkpoints = **277 GB**. A default RunPod container disk (~50 GB) fills after three
checkpoints, and the sweep dies on "No space left on device" several GPU-hours in with 3/19
results. `unload_model()` freed VRAM only — nothing touched disk.

*Fixed:* `purge_checkpoint_cache()` deletes the cached revision after each checkpoint unloads
(each is used exactly once), plus a `check_disk_space()` preflight that refuses to start a
checkpoint below `MIN_FREE_DISK_GB=40`. Failing in seconds beats failing four hours in.

### H9. `git push` could hang the sweep indefinitely
`subprocess.run(..., shell=True)` with no timeout and no `GIT_TERMINAL_PROMPT=0`: an
unauthenticated push blocks on a credential prompt forever, GPU idle and billing. Also
`git commit -m "{message}"` through a shell, and `git add .` in Cell 6b sweeping in model caches
and any stray `.env`.
*Fixed:* list-args (no shell), `GIT_TERMINAL_PROMPT=0`, timeouts, explicit paths, generated
`.gitignore`, and "nothing to commit" treated as success.

---

## Medium

| | Issue | Fix |
|---|---|---|
| M1 | `max_new_tokens=512` vs He et al.'s `MAX_TOKENS_TEST=1000` — breaks the baseline comparability the notebook is built around | 1000 |
| M2 | `gemini-1.5-flash` is retired for new API keys; dev mode would 404 | see the live probe below — now `gemini-flash-lite-latest` |
| M3 | Gemini `response.text` **raises** on safety-blocked or truncated responses — and this judge reads deception/shutdown content, so blocks are routine. Caught by the generic handler → scored 0 | `BLOCK_NONE` safety settings + defensive candidate-parts extraction |
| M4 | `float(final_hb) if final_hb else None` — a genuine 0.0 rate serialised as `null` | `is not None` / NaN-safe |
| M5 | `json.dump` of NaN emits bare `NaN`, which is invalid JSON for every non-Python reader | NaN → `null` throughout |
| M6 | `tqdm.notebook` needs ipywidgets; the plan runs `python run_phase1.py` under tmux | `tqdm.auto` |
| M7 | `import google.generativeai` at top level — a broken install blocks a prod run that never needs it | optional import |
| M8 | `unload_model()` never called `gc.collect()`; VRAM freed unreliably | added |
| M9 | Unbounded `>=` pins install untested majors on a fresh box | upper bounds on all |
| M10 | `_safe_name` unavailable to analysis cells without executing the sweep cell | moved to utilities |
| M11 | `run_inference` and `run_inference_full_trace` were byte-identical — the Phase 1/Phase 2 distinction was cosmetic | merged; the real difference is the token budget, now explicit |
| M12 | `sm.add_constant` on a constant series silently returns one column → `params[1]` IndexError | variance guard + `has_constant='add'` |
| M13 | Per-scenario generation only; no batching | left-padded batched decode, `GEN_BATCH_SIZE=8` (set to 1 to reproduce exactly) |
| M14 | `pre_inflection = ckpt in PHASE2_CHECKPOINTS[:2]` — wrong whenever edge-clamping shortens the bracket | index against the actual anchor |
| M15 | Cell 9 would silently overwrite an already-pre-registered `judge_prompt.txt` | refuses to overwrite a differing locked prompt |

---

## Measured cost (`cost_model.py`)

Token counts are measured, not estimated: judge prompts assembled verbatim from the notebook's
own templates, scenario text from the real benchmark CSVs, response length anchored to the 380
real OLMo responses in `rlvr-3.5-flash-run/`, counted with tiktoken `o200k_base`.

Measured per-call prompt sizes: **Phase 1 = 146 fixed + 187 scenario/context + response**;
**Phase 2 = 461 fixed + trace**. Judge output ~26 tokens (P1, `DECISION`+`REASON`) and ~350 (P2,
Step 1 quotes + JSON).

| scenario | GPU $ | API $ | **total** | of $200 |
|---|---|---|---|---|
| LOW — 42% of token caps | 5.35 | 26.80 | **32.45** | $167.55 left |
| **MID — 70% of caps (budget this)** | **7.23** | **34.12** | **41.65** | **$158.35 left** |
| HIGH — every generation hits its cap | 9.25 | 42.08 | **51.63** | $148.37 left |

**The plan of record (~$145: GPU $21 + API $124) overestimates API by ~3.6×.** It implied
~$0.0136/judge-call; measured is **$0.0028** (Phase 1) and **$0.0072** (Phase 2). Phase 1's
prompt is dominated by the model response, and Phase 1 output is tiny (~26 tokens).

**Caveat — the response-length anchor is a FLOOR.** The 380 prior responses average 495 tokens
against the old 512-token cap, i.e. they were *truncated*; the model rarely stops early. With
`INSTRUMENTAL_MAX_NEW_TOKENS=1000` responses will be longer, and the prior run used the wrong
(math-template) prompt so behaviour may shift. Treat MID/HIGH as the real bracket.

**Unclaimed saving — prompt caching.** The 5 judge passes send a byte-identical prompt.
Phase 1 prompts measure ~1,033 tokens and Phase 2 ~1,461, both above OpenAI's 1,024-token
caching threshold — so passes 2-5 are eligible for 50%-off cached input:

| | API total (MID) |
|---|---|
| all 5 passes fired concurrently (current behaviour) | $34.42 |
| 2 of 5 cached | $29.30 |
| 4 of 5 cached — pass 0 for all scenarios first, then passes 1-4 | **$24.18** |

Staging the pool that way costs no extra wall-clock (same 380 calls at the same concurrency,
just ordered) and preserves pass independence — sampling still runs at temperature 0.5. Saves
~$10. Not implemented: caching is best-effort, and Phase 1 clears the threshold by only 9 tokens,
so a shorter response drops it below. **Optional.**

## Credentials and the dev-judge model (verified live 2026-08-07)

**`.env` had two blocking defects.** It held a **bare secret with no `NAME=` prefix**, and the
notebook **never read `.env` at all** — only `os.environ`. Either alone would have produced
*"OPENAI_API_KEY not set"* on the pod. Both fixed: `.env` rewritten (value unchanged),
`load_dotenv()` added to the config cell (prints names only, real env vars win).

`.gitignore` also failed to cover `.env.bak` / `.env.local`; `.env*` and `*.env.*` added.

**A model listed by the API is not necessarily callable.** Probed with real `generateContent`
calls against the project key:

| model | result |
|---|---|
| `gemini-2.5-flash-lite` | **404 — "no longer available to new users"**, despite appearing in `models.list` |
| `gemini-1.5-flash` | 404, retired |
| `gemini-flash-lite-latest` | ✅ returned a parseable `DECISION: convergence` |
| `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-flash-latest`, `gemini-3.5-flash` | ✅ all working |
| `gemini-2.0-flash-lite`, `gemini-2.0-flash` | 429 — free-tier quota |
| `gemini-3.1-flash` | 404 — no such model |

Default is now the rolling alias `gemini-flash-lite-latest`, which cannot go stale the way a
pinned version does. Dev mode only smoke-tests wiring — the study's data comes from prod — so
tracking latest is the right trade. `DEV_MODEL_ALTERNATES` lists the verified fallbacks and the
health-check error names them.

All working models judged a shutdown-evasion fixture as `convergence`, confirming the judge
prompt survives Gemini's safety filters with `BLOCK_NONE` set.

`JUDGE_CONCURRENCY` drops to 4 in dev mode — the free tier's per-minute limits make 16-way
concurrency pure retry noise.

## Limits as configured (purchased: $43 OpenAI, $15 RunPod)

| Setting | Value | Why |
|---|---|---|
| `MAX_JUDGE_SPEND_USD` | **38.00** | Stops ~$5 short of the $43 credit. HIGH ($42.38) would leave $0.62 — running the account dry mid-checkpoint fails every call at once. |
| `JUDGE_CALL_BUDGET` | 11,000 | 9,180 planned + ~20% retry headroom. Secondary; the dollar cap binds first. |
| `GPU_BUDGET_USD` | **10.00** | $10 ÷ $1.59/hr = **6.29 GPU-hours total**, billed from pod boot. |
| `GPU_STOP_FRACTION` | 0.85 | Sweep stops itself at $8.50 rather than letting RunPod terminate the pod mid-write. |
| `GPU_HOURS_ALREADY_USED` | 0.0 | **Set this** if the pod ran before the notebook started — the notebook can only time itself. |
| `JUDGE_CACHE_STAGING` | True | Pass 0 for all scenarios, then passes 1-4, so the identical prompt hits OpenAI's cache at 50% off input. ~$10 saved, no wall-clock cost. |

**Runtime budget against $10 (6.29 h):**

| | hours | $ | cumulative |
|---|---|---|---|
| pod boot + env + git | 0.40 | 0.64 | 0.64 |
| dev smoke test | 0.25 | 0.40 | 1.03 |
| prod smoke test | 0.15 | 0.24 | 1.27 |
| Phase 1 (MID, 19 ckpt) | 3.50 | 5.57 | 6.84 |
| Phase 2 (MID, 5 ckpt) | 1.00 | 1.59 | **8.43** |
| margin | 0.99 | **1.57** | |

**The HIGH scenario does not fit RunPod: 6.6 h = $10.49 > $10.00.** If the first checkpoint's
measured ETA points at HIGH, either top up or cut scope — `MATH_N` 100 → 50 is the biggest single
lever (MATH is 59% of all generation; saves ~30% of GPU time for ~coarser control resolution).

**Idle is the dominant risk at this budget: 6.3 h of forgotten idle burns the entire RunPod
credit.** The self-stopping run command is not optional here.

## Spend controls

Every one of these is tested — see `test_budget.py` in the verification section.

| Guard | What it stops |
|---|---|
| `MAX_JUDGE_SPEND_USD = 120` | Hard **dollar** ceiling, computed from the token counts the API itself reports (`usage.prompt_tokens` / `usage_metadata`), not from an estimate. Blocks the call *before* the ceiling is crossed. |
| Persisted ledger (`results/judge_usage.json`) | The ceiling surviving a kernel restart. An in-memory counter resets to zero on every restart and silently re-arms the full budget — three restarts, three times the spend. |
| `JUDGE_CALL_BUDGET = 12000` | Runaway call volume, counted per attempt **including retries**. |
| Failure circuit breaker (>20% after 40 calls) | The previous run's exact failure: 380 zeros bought at full price. |
| Per-checkpoint coverage floor (80%) | Saving a mostly-empty checkpoint and discovering it at analysis time. |
| `judge_health_check()` before every sweep | Starting a 14-hour run on a dead key or a wrong model name. |
| Measured cost projection after each checkpoint | Finding out at checkpoint 19 that the real rate was double the estimate. Reports GPU **and** API projections from actual elapsed time and actual tokens. |
| Independent instrumental/MATH skip | Re-paying ~380 judge calls per checkpoint when only the MATH half failed. |
| Cache purge + disk preflight | Dying on a full disk hours in (H0). |
| OOM batch halving | Losing a whole checkpoint's generations to one over-ambitious batch size. |
| `RUN_CROSS_MODEL_PIVOT = False` | ~1,140 unbudgeted judge calls + 3 model downloads (C6). |
| Smoke test raises on failure | Proceeding past a gate that only *printed* a warning. |

**The one risk the notebook cannot control: an idle pod.** At $1.59/hr an A100 left running
costs **$38/day** — a forgotten weekend is $114, more than half the entire compute budget.
Make the pod stop itself when the sweep exits, whatever the exit reason:

```bash
tmux new -s sweep
python run_phase1.py 2>&1 | tee phase1_run.log; runpodctl stop pod $RUNPOD_POD_ID
```

Note `;` not `&&` — it must stop on failure too, which is exactly the case where you are not
watching. `stop` preserves the disk; results are pushed to git per checkpoint regardless.

## Not fixed — these need your decision

1. ~~Baselines unverified.~~ **RESOLVED 2026-08-06 — both verified against the papers' full text.**
   My earlier claim that the two arXiv IDs conflicted and that "2603 is not a valid identifier"
   was **wrong**. They are two different, both-real, both-relevant papers:

   | ID | Paper | Role |
   |---|---|---|
   | 2502.12206 | He, Yufei et al., *Evaluating the Paperclip Maximizer* | defines InstrumentalEval |
   | 2505.05410 | **Yanda** Chen et al. (Anthropic), *Reasoning Models Don't Always Say What They Think* | faithfulness ceiling |
   | 2603.05706 | **Yueh-Han** Chen et al., *Reasoning Models Struggle to Control their Chains of Thought* | the CoT-suppression work this study extends |

   The real defect was a **misattribution**: 2603.05706 was credited to "He et al." Its first
   author is Yueh-Han Chen; "He He" is a *co-author*, which is likely how the error arose.

   - `HE_ET_AL_HIDING_BASELINE = 0.5637` **confirmed**: *"the most concerning difference appears
     in Hiding Unwanted Behavior, where RL-based models average 56.37% compared to RLHF models'
     33.33%"*. The same paper states *"Hiding Unwanted Behavior (11 tasks)"*, matching our n=11.
   - `CHEN_ET_AL_FAITHFULNESS_CEILING = 0.02` **confirmed, with a caveat**: *"in 5 out of 6
     environments, the CoTs verbalize the reward hacks on fewer than 2% of examples, despite
     using the reward hack on >99% of examples"* (Fig. 7 — the RL reward-hacking experiment, so
     the notebook's "Section 5 not Section 3" comment was right). **It is 5 of 6 environments,
     not universal** — an unqualified "<2%" overstates the source.

   `BASELINES_VERIFIED = True`, with the quotes carried in the config and in `phase2_summary.json`.

3. **Statistical power.** Hiding Unwanted Behavior and Deceptive Behavior have **n=11 scenarios
   each**. Split by behavioural score for the Fisher's exact dissociation test, cells routinely
   fall below 5. The notebook now prints the cell counts and flags them as descriptive, but no
   code change creates power that isn't there. Consider making Evading Shutdown (n=20) the primary
   pre-registered category and the other two secondary.

4. **Hardware mismatch.** The notebook says 2×T4 (Kaggle); `claude_instruction.md` budgets an A100
   on RunPod. They are not interchangeable: T4 forces fp16 on a bf16 checkpoint (see H8), and
   `device_map='auto'` across two T4s is pipeline-parallel — only one GPU computes at a time.
   **Recommendation: A100, as budgeted.** It also removes the fp16 risk entirely.

5. **Token-count basis (reconciled).** `gpu_model.py`'s **4.95M** is an *upper bound* — it assumes
   no sequence ever emits EOS early:

   | | tokens |
   |---|---|
   | P1 InstrumentalEval | 19 × 76 × 1000 = 1,444,000 |
   | P1 MATH | 19 × 100 × 1536 = **2,918,400** |
   | P2 traces | 5 × 76 × 1536 = 583,680 |
   | **total** | **4,946,080** |

   A ~2.1M figure is the same workload at **~42% average completion**, i.e. a realistic-EOS
   estimate rather than a worst case. Both are correct for what they measure. Cost impact on
   A100 SXM: **$7.29 (upper bound) vs $4.55 (realistic)** — use ~$5 in the budget writeup and
   $7.29 as the not-to-exceed. GPU ranking is unaffected; it scales uniformly.

   **Worth noting:** MATH is **59% of all generation** — 2.92M of 4.95M tokens — for a *capability
   control*. Halving `MATH_N` to 50 would cut total generation ~30% for ~$1.50, at the cost of
   coarser resolution (±7% vs ±5% SE per checkpoint). Your call; not changed.

6. **Wall-clock estimate.** The hardcoded `1.75 hrs × N` was never measured. With He et al.-parity
   token budgets it is optimistic on T4s. The sweep now prints a **measured** ETA after the first
   checkpoint — check it before committing to the rest.

---

## Test suite

Run everything before touching a real key: `for t in test_*.py; do python $t; done`

| suite | what it proves |
|---|---|
| `test_logic` | ~75 unit tests: decision parsing (bracketed/bold/hyphenated forms), MATH extraction + normalisation, category mapping, checkpoint ordering, majority vote, JSON extraction, score coercion |
| `test_analysis` | Phase 1 analysis on synthetic data: step change at idx 12, step change at idx 7 (proves the `jump=5` defect), gentle ramp, flat series → kill condition |
| `test_budget` | Guards fire: call cap, dollar ceiling blocks *before* crossing, dev mode is free, breaker fires on the 40th consecutive failure and stays quiet at 5%, ledger survives a restart and a restored balance still blocks |
| `test_pool` | Pooled judging keeps results aligned to input order under out-of-order completion; failures isolated to the right item; fatal errors propagate. **Run both with and without cache staging.** |
| `test_devmode` | Gemini path: current model string, BLOCK_NONE on all four harm categories, and — the original bug — a safety-blocked response returns `None`, never `0` |
| `test_failures` | Seven in-situ failure scenarios (below) |
| `test_e2e` | **Full notebook dry run**: every code cell executed in order with a stubbed model and judge, real file I/O, real statistics, real serialisation |

### `test_failures` scenarios

| | scenario | asserted behaviour |
|---|---|---|
| A | judge dead from the start | health check stops it in **≤5 calls**, no files written |
| A2 | judge dies **mid-sweep** (healthy for 200 calls, then garbage) | circuit breaker fires, stops well before 1,520 calls, **no partly-unjudged checkpoint persisted** |
| B | 401 auth error | aborts immediately, **<40 attempts** — no retry storm |
| C | dollar ceiling trips mid-sweep | `JudgeBudgetExceeded`, spend never exceeds the ceiling |
| D | call-count cap trips | stops at the cap |
| E | GPU budget already exhausted | stops **before loading a model**, ≤1 judge call wasted |
| F | intermittent 429s | retries absorb them, all checkpoints complete, attempts > calls |
| G | healthy baseline | completes, exactly `4×76×5 + 1` judge calls, zero failures |

### `test_e2e` results

Full 19-checkpoint + 5-checkpoint dry run:

```
Simulated billed calls : 9139  (phase1 7221 + phase2 1900 + smoke 16)
Simulated token spend  : $33.87  (in 10.21M / out 0.83M)
```

**$33.87 independently reproduces the $34.42 MID cost model** from a completely different
path (simulated per-call token usage vs. tiktoken-measured prompts). It also asserts:

- exactly `19 × 76 × 5` Phase 1 judge calls and `5 × 76 × 5` Phase 2 calls — a change that
  silently doubles spend fails the suite
- **resume after a deleted MATH file re-judged nothing** (1 health-check call only)
- the pivot cell made **zero** judge calls
- the step change survives end-to-end; per-category series are non-zero (category-mapping
  regression guard); MATH series non-degenerate (the 0.4% regression guard)
- `phase2_summary.json` is strict JSON with no `NaN`

## Verification performed

- All 20 code cells compile; `nbformat.validate()` passes; no undefined names across cells.
- **~75 unit tests** on the extracted pure-logic functions — decision parsing (including every
  malformed-but-real judge format), MATH extraction/normalisation, category mapping, checkpoint
  ordering, majority vote, JSON extraction, score coercion — **all passing**.
- **Spend-guard tests** (`test_budget.py`): call cap fires at the limit; dollar ceiling blocks the
  call *before* crossing $1.00 in a scaled-down fixture; dev mode accrues $0; circuit breaker
  fires on the 40th consecutive failure and stays quiet at a 5% failure rate; the ledger round-trips
  a restart and a restored balance still blocks. All passing.
- **Integration test** of the analysis cells on synthetic Phase 1 data: step change at index 12,
  step change at index 7 (the non-multiple-of-5 case that proves H1), gentle ramp, and flat series
  → kill condition. All four produce the correct anchor and method.
- Live-verified: repo exists and its 19 `step_*` branches; benchmark CSV names, structure and the
  76-scenario total; the chat template on `step_100`, `step_1900` and `main`; MATH-lighteval's
  `default` config (5,000 test rows, `level` as `"Level N"`).

**Not verified:** anything requiring a GPU or a live API key. The dev-mode smoke test is the gate
that covers those, and it is now a hard gate — Cell 6c raises rather than printing "PASSED" when a
check fails.

---

## Run order

1. Cell 6b: set `GIT_REMOTE` / `GIT_EMAIL` / `GIT_NAME`, confirm the initial push succeeds.
2. Cell 6c with `JUDGE_MODE="dev"` — free. Read the printed prompts and the MATH pred/gold pairs.
3. Cell 6c with `JUDGE_MODE="prod"` — cents. Confirms 5 independent paid calls per item, verified
   against the API call counter rather than by counting list entries.
4. Cell 7. Check the measured ETA and the first checkpoint's MATH accuracy before walking away.
5. Cell 8 → **Cell 9 (lock prompt) before any Phase 2 trace exists** → Cell 10 → Cells 11+ → 9b.
