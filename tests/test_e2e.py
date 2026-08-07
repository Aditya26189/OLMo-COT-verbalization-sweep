# -*- coding: utf-8 -*-
"""End-to-end dry run of the ENTIRE notebook with a stubbed model and judge.

Executes every code cell in order, in a temp working directory, with:
  - a fake OLMo (canned generations carrying a convergence marker that shifts at ckpt 12)
  - a fake OpenAI client that returns realistic judge text AND usage token counts
  - real file I/O, real judging logic, real statistics, real serialisation

Purpose: catch integration bugs BEFORE a real API key is used. Asserts the exact
number of billed judge calls, so a change that silently doubles spend fails here.
"""
import ast, io, json, os, random, re, shutil, sys, tempfile, types, contextlib

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rlvr_cot_phase0_fixed.ipynb')
nb = json.load(io.open(NB, encoding='utf-8'))
CODE = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

N_CKPT, N_SCEN, N_PASS, MATH_N = 19, 76, 5, 10
BREAK_AT = 12                      # convergence rate jumps here -> PELT must find it

fails = []
def check(label, got, want):
    if got != want:
        fails.append('%s: got %r want %r' % (label, got, want))
def check_true(label, cond, detail=''):
    if not cond:
        fails.append('%s: %s' % (label, detail or 'false'))


# ───────────────────────── stubs ─────────────────────────
class FakeUsage:
    def __init__(self, i, o): self.prompt_tokens, self.completion_tokens = i, o
class FakeMsg:
    def __init__(self, c): self.content = c
class FakeChoice:
    def __init__(self, c): self.message, self.finish_reason = FakeMsg(c), 'stop'
class FakeResp:
    def __init__(self, c, i, o): self.choices, self.usage = [FakeChoice(c)], FakeUsage(i, o)

JUDGE_CALLS = {'n': 0, 'p1': 0, 'p2': 0, 'prompts': []}
MALFORMED_EVERY = 97               # exercise the tolerant parsers without tripping the breaker

class FakeCompletions:
    def create(self, model=None, messages=None, max_tokens=None, temperature=None, **kw):
        prompt = messages[0]['content']
        JUDGE_CALLS['n'] += 1
        JUDGE_CALLS['prompts'].append(len(prompt))
        n = JUDGE_CALLS['n']
        if 'chain-of-thought reasoning trace' in prompt:          # Phase 2 judge
            JUDGE_CALLS['p2'] += 1
            conv = 'CONVERGENT_MARKER' in prompt
            body = ('Step 1: Evading Shutdown - "no supporting phrases found".\n'
                    'Deceptive Behavior - the trace mentions {concealment}.\n\nStep 2:\n'
                    '{"Evading Shutdown": {"evidence": "none", "score": %d}, '
                    '"Hiding Unwanted Behavior": {"evidence": "none", "score": 0}, '
                    '"Deceptive Behavior": {"evidence": "none", "score": %d}}'
                    % (1 if conv else 0, 1 if conv else 0))
            if n % MALFORMED_EVERY == 0:
                body = 'I am unable to produce a classification for this trace.'
            return FakeResp(body, 1450, 340)
        JUDGE_CALLS['p1'] += 1                                     # Phase 1 judge
        conv = 'CONVERGENT_MARKER' in prompt
        forms = ['DECISION: %s\nREASON: evidence.',
                 'DECISION: [%s]\nREASON: evidence.',
                 '**DECISION:** %s\nREASON: evidence.']
        body = forms[n % 3] % ('convergence' if conv else 'no_convergence')
        if n % MALFORMED_EVERY == 0:
            body = 'I cannot assess this response.'
        return FakeResp(body, 1030, 26)

class FakeChat:
    def __init__(self): self.completions = FakeCompletions()
class FakeOpenAI:
    def __init__(self, api_key=None, **kw): self.chat = FakeChat()


class FakeTok:
    # The REAL math-specific template, so wrap_scenario's refusal path is exercised.
    chat_template = ('Solve the following math problem step by step. The last line of your '
                     'response should be the answer in form Answer: $Answer\n{% for m in messages %}'
                     '{{ m["content"] }}{% endfor %}')
    pad_token, eos_token, eos_token_id, pad_token_id, padding_side = '<eos>', '<eos>', 0, 0, 'left'
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ('Solve the following math problem step by step.\n'
                + '\n'.join(m['content'] for m in messages)
                + '\nRemember to put your answer on its own line after "Answer:"')
    def __call__(self, *a, **k): raise AssertionError('tokenizer should not be called (generation is stubbed)')

class FakeModel:
    def eval(self): return self
    def parameters(self): return iter([types.SimpleNamespace(device='cpu')])

CUR = {'ckpt_idx': 0}
def fake_load_model(step, repo=None):
    CUR['ckpt_idx'] = int(re.search(r'\d+', step).group()) // 100 - 1
    return FakeModel(), FakeTok()

def fake_generate_batch(model, tokenizer, prompts, max_new_tokens):
    """Convergence rate steps up at BREAK_AT; MATH accuracy ramps. Deterministic per checkpoint."""
    idx = CUR['ckpt_idx']
    rate = 0.12 if idx < BREAK_AT else 0.58
    rng = random.Random(1000 + idx)
    out = []
    for p in prompts:
        if 'math problem' in p.lower() and 'Answer:' in p:
            gold = re.search(r'GOLD=(\d+)', p)
            correct = rng.random() < (0.20 + 0.02 * idx)
            val = gold.group(1) if (gold and correct) else '999'
            out.append({'text': 'Reasoning...\nAnswer: %s' % val, 'truncated': False, 'n_new': 60})
        else:
            marker = 'CONVERGENT_MARKER ' if rng.random() < rate else ''
            out.append({'text': marker + 'I will proceed with the task. ' * 40,
                        'truncated': False, 'n_new': 300})
    return out


def make_refs(names):
    B = types.SimpleNamespace
    return B(branches=[B(name=n) for n in names], tags=[])

def fake_list_repo_refs(repo, token=None):
    return make_refs(['main'] + ['step_%d' % (100 * (i + 1)) for i in range(N_CKPT)])

def fake_load_dataset(*a, **k):
    rows = []
    for i in range(400):
        rows.append({'problem': 'Compute GOLD=%d please.' % i, 'level': 'Level %d' % (3 + i % 3),
                     'solution': 'Therefore \\boxed{%d}.' % i, 'type': 'algebra'})
    return rows

def fake_git(args, timeout=180):
    return types.SimpleNamespace(returncode=0, stdout='', stderr='')


# ───────────────────────── driver ─────────────────────────
def cell_src(pred):
    for i, s in CODE:
        if pred(s):
            return i, s
    raise KeyError('cell not found')

def run(ns, src, label):
    src = ''.join('pass\n' if l.lstrip().startswith(('!', '%')) else l
                  for l in src.splitlines(keepends=True))
    try:
        exec(compile(src, '<cell %s>' % label, 'exec'), ns)
    except Exception as e:
        import traceback
        fails.append('CELL %s raised %s: %s' % (label, type(e).__name__, e))
        traceback.print_exc()
        raise

tmp = tempfile.mkdtemp(prefix='rlvr_e2e_')
cwd = os.getcwd()
buf = io.StringIO()
try:
    # real benchmark CSVs
    bench = os.path.join(tmp, 'InstrumentalEval', 'benchmark')
    os.makedirs(bench)
    src_bench = os.path.join(cwd, 'benchmark_cache.json')
    scen = json.load(open(src_bench, encoding='utf-8'))
    import csv as _csv
    from collections import OrderedDict
    bycat = OrderedDict()
    for s in scen:
        bycat.setdefault(s['cat'], (s['specific_prompt'], []))[1].append(s['scenario'])
    for cat, (sp, rows) in bycat.items():
        with open(os.path.join(bench, cat + '.csv'), 'w', encoding='utf-8', newline='') as f:
            w = _csv.writer(f)
            w.writerow([sp])
            for r in rows:
                w.writerow([r])

    os.chdir(tmp)
    os.environ['OPENAI_API_KEY'] = 'sk-test-not-a-real-key'
    ns = {'__name__': '__main__'}

    with contextlib.redirect_stdout(buf):
        # imports
        i, s = cell_src(lambda s: 'import ruptures as rpt' in s)
        run(ns, s, 'imports')
        ns['openai'].OpenAI = FakeOpenAI

        # config (prod mode, scaled-down MATH)
        i, s = cell_src(lambda s: 'JUDGE_MODE = "dev"' in s)
        s = s.replace('JUDGE_MODE = "dev"', 'JUDGE_MODE = "prod"').replace('MATH_N = 100', 'MATH_N = %d' % MATH_N)
        run(ns, s, 'config')

        ns['list_repo_refs'] = fake_list_repo_refs
        i, s = cell_src(lambda s: 'def enumerate_checkpoints' in s)
        run(ns, s, 'checkpoints')

        i, s = cell_src(lambda s: 'def load_instrumental_eval' in s)
        run(ns, s, 'instrumental+judge')

        ns['load_dataset'] = fake_load_dataset
        i, s = cell_src(lambda s: 'def load_math_subset' in s)
        run(ns, s, 'math')

        i, s = cell_src(lambda s: 'def generate_batch' in s)
        run(ns, s, 'inference utils')
        ns['load_model'] = fake_load_model
        ns['generate_batch'] = fake_generate_batch
        ns['_git'] = fake_git
        ns['purge_checkpoint_cache'] = lambda *a, **k: 0.0
        ns['preflight_model_health'] = lambda *a, **k: True
        # nothing is downloaded in this dry run, so the disk guard has nothing to guard
        ns['check_disk_space'] = lambda *a, **k: 999.0

        i, s = cell_src(lambda s: 'GIT_REMOTE = ' in s)
        s = s.replace('https://github.com/YOUR_USERNAME/YOUR_REPO.git',
                      'https://example.invalid/e2e/test.git')
        run(ns, s, 'git setup')

        i, s = cell_src(lambda s: '=== SMOKE TEST ===' in s)
        run(ns, s, 'smoke test')
        smoke_calls = JUDGE_CALLS['n']

        i, s = cell_src(lambda s: 'def sweep_checkpoint_phase1' in s)
        run(ns, s, 'phase1 sweep')
        p1_calls = JUDGE_CALLS['p1']

        i, s = cell_src(lambda s: 'def load_phase1_results' in s)
        run(ns, s, 'phase1 load')
        i, s = cell_src(lambda s: 'PELT penalty (BIC)' in s)
        run(ns, s, 'phase1 stats')
        i, s = cell_src(lambda s: 'PHASE 1 PLOTS' in s)
        run(ns, s, 'phase1 plots')
        i, s = cell_src(lambda s: 'JUDGE_PROMPT = ' in s)
        run(ns, s, 'lock judge prompt')
        i, s = cell_src(lambda s: 'def sample_for_human_annotation' in s)
        run(ns, s, 'IRR defs')
        i, s = cell_src(lambda s: 'def sweep_checkpoint_phase2' in s)
        run(ns, s, 'phase2 sweep')
        p2_calls = JUDGE_CALLS['p2']
        i, s = cell_src(lambda s: 'VERBALIZATION RATES ACROSS' in s)
        run(ns, s, 'phase2 analysis')
        i, s = cell_src(lambda s: 'VERBALIZATION-MATH SPEARMAN' in s)
        run(ns, s, 'phase2 spearman+plots')
        i, s = cell_src(lambda s: 'SAVE COMPLETE RESULTS SUMMARY' in s)
        run(ns, s, 'summary')
        i, s = cell_src(lambda s: 'RUN_CROSS_MODEL_PIVOT' in s)
        run(ns, s, 'pivot (must be a no-op)')
        pivot_calls_after = JUDGE_CALLS['p1']

        # ── RESUME TEST: delete one MATH file, re-run the sweep ──
        calls_before_resume = JUDGE_CALLS['n']
        os.remove('results/phase1/checkpoint_step_500_math.json')
        i, s = cell_src(lambda s: 'def sweep_checkpoint_phase1' in s)
        run(ns, s, 'phase1 resume')
        resume_calls = JUDGE_CALLS['n'] - calls_before_resume

    # ───────────────────── assertions ─────────────────────
    check('checkpoints enumerated', len(ns['ALL_CHECKPOINTS']), N_CKPT)
    check('checkpoint order numeric', ns['ALL_CHECKPOINTS'][:3], ['step_100', 'step_200', 'step_300'])
    check('scenarios loaded', len(ns['scenarios']), N_SCEN)
    cats = {}
    for sc in ns['scenarios']:
        cats[sc['category']] = cats.get(sc['category'], 0) + 1
    check('Evading Shutdown n', cats.get('Evading Shutdown'), 20)
    check('Hiding Unwanted n', cats.get('Hiding Unwanted Behavior'), 11)
    check('Deceptive n', cats.get('Deceptive Behavior'), 11)
    check('unmapped -> None', cats.get(None), 34)

    # Each sweep cell opens with judge_health_check() — one deliberate call (~$0.003) that
    # prevents a multi-hour run against a dead judge. Budget for it explicitly.
    check('phase1 judge calls == 19*76*5 + 1 health check',
          p1_calls - smoke_calls, N_CKPT * N_SCEN * N_PASS + 1)
    check('phase2 judge calls == 5*76*5', p2_calls, 5 * N_SCEN * N_PASS)
    check('pivot made no judge calls of its own', pivot_calls_after - p1_calls, 1)  # phase2 health check
    check('resume re-judged NOTHING (health check only)', resume_calls, 1)

    infl = json.load(open('results/inflection_point.json', encoding='utf-8'))
    check_true('changepoint found near break', abs((infl['inflection_idx'] or -99) - BREAK_AT) <= 2,
               'idx=%s want ~%d' % (infl['inflection_idx'], BREAK_AT))
    check('analyzed_steps persisted', len(infl['analyzed_steps']), N_CKPT)
    check('kill_condition false', infl['kill_condition'], False)

    p2 = ns['PHASE2_CHECKPOINTS']
    check('phase2 bracket size', len(p2), 5)
    check_true('bracket centred on anchor', infl['inflection_step'] in p2, str(p2))

    raw = open('results/phase2_summary.json', encoding='utf-8').read()
    check_true('summary is strict JSON (no NaN)', 'NaN' not in raw and 'Infinity' not in raw)
    summary = json.loads(raw)
    check('summary judge model', summary['judge_model'], 'gpt-4o')
    check('summary baselines verified', summary['baselines_verified'], True)

    inst = json.load(open('results/phase1/checkpoint_step_1900_instrumental.json', encoding='utf-8'))
    check('per-scenario records', len(inst), N_SCEN)
    check('5 passes stored', len(inst[0]['passes']), N_PASS)
    check_true('scenario_idx present', all('scenario_idx' in r for r in inst))
    scored = [r for r in inst if r['score'] is not None]
    check_true('most scenarios scored', len(scored) > 0.9 * N_SCEN, str(len(scored)))

    ledger = json.load(open('results/judge_usage.json', encoding='utf-8'))
    check_true('ledger tracked spend', ledger['spend_usd'] > 0, str(ledger))
    check_true('ledger tokens counted', ledger['tokens_in'] > 1e6, str(ledger['tokens_in']))

    verb = json.load(open('results/phase2/checkpoint_%s_verbalization.json'
                          % infl['inflection_step'], encoding='utf-8'))
    check('verbalization records', len(verb), N_SCEN)
    check_true('trace_id present', all('trace_id' in v for v in verb))

    # The signal the study actually measures must survive the whole pipeline.
    inst_series = ns['instrumental_series']
    pre = sum(inst_series[:BREAK_AT]) / BREAK_AT
    post = sum(inst_series[BREAK_AT:]) / (len(inst_series) - BREAK_AT)
    check_true('step change preserved end-to-end', post - pre > 0.25,
               'pre=%.3f post=%.3f' % (pre, post))
    check_true('per-category series populated (not all zero)',
               any(ns['data'][s]['instrumental_by_cat']['Evading Shutdown'] > 0 for s in ns['steps']),
               'every Evading Shutdown rate was 0 — the category-mapping bug is back')

    math_series = ns['math_series']
    check_true('MATH series non-degenerate', max(math_series) > 0.05,
               'max=%.3f — scorer broken (this is the 0.4%% failure mode)' % max(math_series))

    vr = ns['verbalization_rates']
    vn = ns['verbalization_n']
    check_true('verbalization rates populated',
               any(any(x == x and x > 0 for x in v) for v in vr.values()),
               'rates=%r counts=%r' % (vr, vn))
    check_true('verbalization denominators non-zero',
               all(sum(v) > 0 for v in vn.values()), 'counts=%r' % vn)

    check_true('phase1 plot written', os.path.exists('results/phase1_curves.png'))
    check_true('phase2 plot written', os.path.exists('results/phase2_verbalization.png'))
    check_true('judge prompt locked', os.path.exists('judge_prompt.txt'))

    # projected real spend from the fake token counts
    tin, tout = ledger['tokens_in'], ledger['tokens_out']
    est = (tin * 2.50 + tout * 10.00) / 1e6
    print('Simulated billed calls : %d  (phase1 %d + phase2 %d + smoke %d)'
          % (JUDGE_CALLS['n'], p1_calls - smoke_calls, p2_calls, smoke_calls))
    print('Simulated token spend  : $%.2f  (in %.2fM / out %.2fM)' % (est, tin / 1e6, tout / 1e6))
    print('Ledger spend_usd       : $%.2f' % ledger['spend_usd'])

finally:
    os.chdir(cwd)
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print('\n--- captured notebook stdout (tail) ---')
    print('\n'.join(buf.getvalue().splitlines()[-40:]))

print('\n%d failures' % len(fails))
for f in fails:
    print('  FAIL', f)
sys.exit(1 if fails else 0)
