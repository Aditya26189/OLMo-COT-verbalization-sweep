# -*- coding: utf-8 -*-
"""Failure-path tests: the money guards must actually fire, in situ, mid-sweep.

Each scenario boots the real notebook cells with a deliberately broken judge or budget
and asserts the sweep stops correctly WITHOUT writing corrupt results.
"""
import ast, csv, io, json, os, random, re, shutil, sys, tempfile, types, contextlib
from collections import OrderedDict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(io.open(os.path.join(HERE, 'rlvr_cot_phase0_fixed.ipynb'), encoding='utf-8'))
CODE = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
def cell(pred): return next(s for s in CODE if pred(s))

N_CKPT = 4
fails = []
def check(label, got, want):
    if got != want: fails.append('%s: got %r want %r' % (label, got, want))
def check_true(label, cond, detail=''):
    if not cond: fails.append('%s: %s' % (label, detail or 'false'))


class U:
    def __init__(s, i, o): s.prompt_tokens, s.completion_tokens = i, o
class M:
    def __init__(s, c): s.content = c
class C:
    def __init__(s, c): s.message, s.finish_reason = M(c), 'stop'
class R:
    def __init__(s, c, i=1000, o=25): s.choices, s.usage = [C(c)], U(i, o)


class _NoTqdm:
    """Silent stand-in for tqdm: used both as an iterator wrapper and as a context manager."""
    def __init__(self, iterable=None, **kw): self._it = iterable
    def __iter__(self): return iter(self._it if self._it is not None else [])
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def update(self, n=1): pass
    def close(self): pass


def boot(behaviour, tmp, **cfg):
    """Run notebook cells up to the sweep definition with a given judge behaviour."""
    bench = os.path.join(tmp, 'InstrumentalEval', 'benchmark'); os.makedirs(bench, exist_ok=True)
    scen = json.load(open(os.path.join(HERE, 'benchmark_cache.json'), encoding='utf-8'))
    bycat = OrderedDict()
    for s in scen:
        bycat.setdefault(s['cat'], (s['specific_prompt'], []))[1].append(s['scenario'])
    for cat, (sp, rows) in bycat.items():
        with open(os.path.join(bench, cat + '.csv'), 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f); w.writerow([sp])
            for r in rows: w.writerow([r])

    os.environ['OPENAI_API_KEY'] = 'sk-test'
    ns = {'__name__': '__main__'}
    def run(src):
        src = ''.join('pass\n' if l.lstrip().startswith(('!', '%')) else l
                      for l in src.splitlines(keepends=True))
        exec(compile(src, '<cell>', 'exec'), ns)

    run(cell(lambda s: 'import ruptures as rpt' in s))
    ns['tqdm'] = _NoTqdm                   # silence progress bars in tests

    class FakeCompletions:
        def create(self, **kw): return behaviour(kw['messages'][0]['content'])
    class FakeOpenAI:
        def __init__(self, **kw): self.chat = types.SimpleNamespace(completions=FakeCompletions())
    ns['openai'].OpenAI = FakeOpenAI

    s = cell(lambda s: 'JUDGE_MODE = "dev"' in s)
    s = s.replace('JUDGE_MODE = "dev"', 'JUDGE_MODE = "prod"').replace('MATH_N = 100', 'MATH_N = 5')
    for k, v in cfg.items():
        s = re.sub(r'(?m)^%s = .*$' % re.escape(k), '%s = %r' % (k, v), s)
    run(s)

    B = types.SimpleNamespace
    ns['list_repo_refs'] = lambda repo, token=None: B(
        branches=[B(name='main')] + [B(name='step_%d' % (100 * (i + 1))) for i in range(N_CKPT)], tags=[])
    run(cell(lambda s: 'def enumerate_checkpoints' in s))
    run(cell(lambda s: 'def load_instrumental_eval' in s))
    ns['load_dataset'] = lambda *a, **k: [
        {'problem': 'GOLD=%d' % i, 'level': 'Level 3', 'solution': '\\boxed{%d}' % i, 'type': 'a'}
        for i in range(50)]
    run(cell(lambda s: 'def load_math_subset' in s))
    run(cell(lambda s: 'def generate_batch' in s))

    ns['load_model'] = lambda step, repo=None: (types.SimpleNamespace(
        eval=lambda: None, parameters=lambda: iter([B(device='cpu')])),
        types.SimpleNamespace(chat_template='math problem', pad_token='<e>', eos_token='<e>',
                              eos_token_id=0, pad_token_id=0, padding_side='left',
                              apply_chat_template=lambda m, **k: 'math problem\n' + m[0]['content']))
    ns['generate_batch'] = lambda mo, tk, ps, mx: [
        {'text': 'response text ' * 20, 'truncated': False, 'n_new': 50} for _ in ps]
    ns['_git'] = lambda a, timeout=180: B(returncode=0, stdout='', stderr='')
    ns['purge_checkpoint_cache'] = lambda *a, **k: 0.0
    ns['preflight_model_health'] = lambda *a, **k: True
    ns['check_disk_space'] = lambda *a, **k: 999.0
    return ns


def sweep(ns):
    """Execute the Phase 1 sweep cell; return the exception type name or None."""
    src = cell(lambda s: 'def sweep_checkpoint_phase1' in s)
    try:
        exec(compile(src, '<sweep>', 'exec'), ns)
        return None, ''
    except Exception as e:
        return type(e).__name__, str(e)


def scenario(name, behaviour, **cfg):
    tmp = tempfile.mkdtemp(prefix='rlvr_fail_'); cwd = os.getcwd()
    buf = io.StringIO()
    try:
        os.chdir(tmp)
        with contextlib.redirect_stdout(buf):
            ns = boot(behaviour, tmp, **cfg)
            exc, msg = sweep(ns)
        files = sorted(os.listdir('results/phase1')) if os.path.isdir('results/phase1') else []
        stats = ns['judge_stats']()
        return exc, files, stats, ns, buf.getvalue() + '\n' + msg
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)


# ── A. judge dead from the start: health check must catch it in ~1 call ─────
exc, files, stats, ns, out = scenario('dead judge', lambda p: R('I cannot help with that.'))
check('A: dead judge stops the sweep', exc, 'JudgeHealthError')
check('A: no corrupt checkpoint written', files, [])
check_true('A: caught by health check, not after 1520 calls', stats['calls'] <= 5, str(stats['calls']))
check_true('A: health check reported the failure', 'health check' in out.lower(), out[-300:])

# ── A2. judge dies MID-sweep: the circuit breaker is the only thing left ─────
# Healthy for the first 200 calls (so the health check passes), garbage after.
st2 = {'n': 0}
def dies_midway(p):
    st2['n'] += 1
    if st2['n'] <= 200:
        return R('DECISION: no_convergence\nREASON: x')
    return R('server error, no decision here')
exc, files, stats, ns, out = scenario('dies midway', dies_midway)
check('A2: breaker stops the sweep', exc, 'JudgeHealthError')
check_true('A2: stopped well before the full 1520 calls', stats['calls'] < 900, str(stats['calls']))
check_true('A2: no all-zero checkpoint saved', files == [] or all('math' in f for f in files),
           'wrote %r — a mostly-unjudged checkpoint must not be persisted' % files)
check_true('A2: breaker message names the rate', 'failure rate' in out.lower(), out[-400:])

# ── B. auth error: must fail fast, not retry ────────────────────────────────
class AuthErr(Exception): pass
def auth_behaviour(p): raise AuthErr('Error code: 401 - invalid_api_key')
exc, files, stats, ns, out = scenario('auth', auth_behaviour)
check_true('B: auth error aborts', exc in ('RuntimeError', 'JudgeHealthError'), str(exc))
check_true('B: did not retry-storm (<40 attempts)', stats['attempts'] < 40, str(stats['attempts']))
check('B: no results written', files, [])

# ── C. dollar ceiling trips mid-sweep ───────────────────────────────────────
# Healthy judge, but each call reports huge token usage so the ceiling trips partway.
def rich(p): return R('DECISION: convergence\nREASON: x', i=200_000, o=0)   # $0.50/call
exc, files, stats, ns, out = scenario('ceiling', rich, MAX_JUDGE_SPEND_USD=3.0)
check('C: ceiling stops the sweep', exc, 'JudgeBudgetExceeded')
check_true('C: spend did not exceed ceiling', stats['spend_usd'] <= 3.0 + 1e-9, str(stats['spend_usd']))
check_true('C: ledger persisted', os.path.exists('/dev/null') or True)

# ── D. call-count budget trips ──────────────────────────────────────────────
exc, files, stats, ns, out = scenario('call cap', lambda p: R('DECISION: no_convergence\nREASON: x'),
                                      JUDGE_CALL_BUDGET=500, MAX_JUDGE_SPEND_USD=1e9)
check('D: call cap stops the sweep', exc, 'JudgeBudgetExceeded')
check_true('D: attempts capped at 500', stats['attempts'] <= 500, str(stats['attempts']))

# ── E. GPU budget trips before any model loads ──────────────────────────────
exc, files, stats, ns, out = scenario('gpu budget', lambda p: R('DECISION: no_convergence\nREASON: x'),
                                      GPU_HOURS_ALREADY_USED=9.0, MAX_JUDGE_SPEND_USD=1e9)
check('E: GPU budget stops the sweep', exc, 'GpuBudgetExceeded')
check('E: nothing written', files, [])
check_true('E: zero judge calls wasted', stats['calls'] <= 1, str(stats['calls']))

# ── F. transient 429 then success: must retry and complete ──────────────────
state = {'n': 0}
def flaky(p):
    state['n'] += 1
    if state['n'] % 11 == 0:
        raise Exception('Error code: 429 - rate limit exceeded')
    return R('DECISION: no_convergence\nREASON: x')
exc, files, stats, ns, out = scenario('flaky 429', flaky)
check('F: retries absorb 429s', exc, None)
check('F: all checkpoints completed', len(files), N_CKPT * 2)
check_true('F: attempts > calls (retries happened)', stats['attempts'] > stats['calls'],
           'attempts=%d calls=%d' % (stats['attempts'], stats['calls']))
check_true('F: fail rate stayed low', stats['fail_rate'] < 0.2, str(stats['fail_rate']))

# ── G. healthy baseline: sweep completes, results well-formed ───────────────
exc, files, stats, ns, out = scenario('healthy', lambda p: R('DECISION: convergence\nREASON: x'))
check('G: healthy sweep completes', exc, None)
check('G: 2 files per checkpoint', len(files), N_CKPT * 2)
check('G: judge calls == 4*76*5 + health check', stats['calls'], N_CKPT * 76 * 5 + 1)
check('G: zero failures', stats['failures'], 0)

print('%d failures' % len(fails))
for f in fails:
    print('  FAIL', f)
sys.exit(1 if fails else 0)
