# -*- coding: utf-8 -*-
"""Prove the spend guards actually fire: budget cap, dollar ceiling, circuit breaker, persistence."""
import ast, io, json, os, re, sys, tempfile, threading, time
from pathlib import Path

NB = r'c:\Users\LawLight\Desktop\rlvr-cot\rlvr_cot_phase0_fixed.ipynb'
nb = json.load(io.open(NB, encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

# Pull the judge-accounting machinery out of the InstrumentalEval cell.
CELL = next(s for s in cells if 'def _reserve_judge_call' in s)
WANT = {'JudgeBudgetExceeded', 'JudgeHealthError', '_judge_lock', '_judge_calls',
        '_judge_attempts', '_judge_failures', '_judge_tokens_in', '_judge_tokens_out',
        '_judge_breaker_tripped', '_load_usage_ledger', '_session_spend_usd', 'judge_stats',
        'save_usage_ledger', 'reset_judge_stats', '_record_tokens', '_reserve_judge_call',
        '_record_judge_outcome', '_PRIOR_ATTEMPTS', '_PRIOR_CALLS', '_PRIOR_TOK_IN',
        '_PRIOR_TOK_OUT', '_PRIOR_SPEND'}

fails = []
def check(label, got, want):
    if got != want:
        fails.append('%s: got %r want %r' % (label, got, want))

def fresh(tmp, **cfg):
    """Build a namespace with the notebook's accounting code and a given config."""
    ns = {'json': json, 'os': os, 'time': time, 'threading': threading, 'Path': Path, 're': re}
    ns.update({'JUDGE_MODE': 'prod', 'JUDGE_MODEL': 'gpt-4o',
               'JUDGE_CALL_BUDGET': 100, 'MAX_JUDGE_SPEND_USD': 1.00,
               'JUDGE_PRICE_IN_PER_1M': 2.50, 'JUDGE_PRICE_OUT_PER_1M': 10.00,
               'JUDGE_FAIL_RATE_ABORT': 0.20, 'JUDGE_FAIL_MIN_SAMPLES': 40,
               'JUDGE_USAGE_PATH': os.path.join(tmp, 'judge_usage.json')})
    ns.update(cfg)
    for node in ast.parse(CELL).body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', None)
        if isinstance(node, ast.Tuple):
            name = None
        # multi-target assign like _PRIOR_A, _PRIOR_B = ...
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            name = '_PRIOR_ATTEMPTS'
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in WANT:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<nb>', 'exec'), ns)
        elif isinstance(node, ast.Assign) and (name in WANT):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<nb>', 'exec'), ns)
        elif isinstance(node, ast.If) and 'PRIOR' in ast.dump(node):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<nb>', 'exec'), ns)
    return ns


# ---------- 1. call-count budget fires ----------
tmp = tempfile.mkdtemp()
ns = fresh(tmp, JUDGE_CALL_BUDGET=10, MAX_JUDGE_SPEND_USD=1e9)
n = 0
try:
    for _ in range(50):
        ns['_reserve_judge_call']()
        n += 1
except Exception as e:
    check('call cap type', type(e).__name__, 'JudgeBudgetExceeded')
check('call cap stops at budget', n, 10)

# ---------- 2. dollar ceiling fires on real token usage ----------
tmp = tempfile.mkdtemp()
ns = fresh(tmp, JUDGE_CALL_BUDGET=10**6, MAX_JUDGE_SPEND_USD=1.00)
# 1M in-tokens = $2.50 > $1.00 ceiling. 100k in-tokens = $0.25 each.
spent, err = 0, None
try:
    for _ in range(20):
        ns['_reserve_judge_call']()
        ns['_record_tokens'](100_000, 0)   # $0.25 per call
        spent += 1
except Exception as e:
    err = type(e).__name__
check('usd ceiling type', err, 'JudgeBudgetExceeded')
check('usd ceiling stops after 4 calls', spent, 4)   # 4x$0.25 = $1.00; the 5th reserve is refused
# Guard blocks BEFORE overspending: total lands exactly on the ceiling, never past it.
check('usd accounted', round(ns['judge_stats']()['spend_usd'], 2), 1.00)

# ---------- 3. dev mode is free (no dollar guard trip) ----------
tmp = tempfile.mkdtemp()
ns = fresh(tmp, JUDGE_MODE='dev', MAX_JUDGE_SPEND_USD=0.01, JUDGE_CALL_BUDGET=10**6)
for _ in range(50):
    ns['_reserve_judge_call']()
    ns['_record_tokens'](100_000, 100_000)
check('dev mode spend is 0', ns['judge_stats']()['spend_usd'], 0.0)

# ---------- 4. circuit breaker fires on sustained failures ----------
tmp = tempfile.mkdtemp()
ns = fresh(tmp, JUDGE_CALL_BUDGET=10**6, MAX_JUDGE_SPEND_USD=1e9,
           JUDGE_FAIL_MIN_SAMPLES=40, JUDGE_FAIL_RATE_ABORT=0.20)
ok_n, err = 0, None
try:
    for _ in range(200):
        ns['_record_judge_outcome'](False)     # every call fails, like the previous run
        ok_n += 1
except Exception as e:
    err = type(e).__name__
check('breaker type', err, 'JudgeHealthError')
# Raises ON the 40th call (JUDGE_FAIL_MIN_SAMPLES), so 39 complete before it trips.
check('breaker fires on the 40th call', ok_n, 39)

# a healthy judge must NOT trip it
tmp = tempfile.mkdtemp()
ns = fresh(tmp, JUDGE_CALL_BUDGET=10**6, MAX_JUDGE_SPEND_USD=1e9)
tripped = False
try:
    for i in range(300):
        ns['_record_judge_outcome'](i % 20 != 0)   # 5% failure rate
except Exception:
    tripped = True
check('breaker quiet on healthy judge', tripped, False)

# ---------- 5. ledger persists across "restart" ----------
tmp = tempfile.mkdtemp()
ns = fresh(tmp, JUDGE_CALL_BUDGET=10**6, MAX_JUDGE_SPEND_USD=10.0)
for _ in range(4):
    ns['_reserve_judge_call']()
    ns['_record_tokens'](100_000, 0)          # $0.25 each -> $1.00
ns['save_usage_ledger']()
check('ledger written', os.path.exists(os.path.join(tmp, 'judge_usage.json')), True)

ns2 = fresh(tmp, JUDGE_CALL_BUDGET=10**6, MAX_JUDGE_SPEND_USD=10.0)   # simulate restart
check('prior spend restored', round(ns2['judge_stats']()['spend_usd'], 2), 1.00)
check('prior attempts restored', ns2['judge_stats']()['attempts'], 4)

# and the restored spend counts against the ceiling
ns3 = fresh(tmp, JUDGE_CALL_BUDGET=10**6, MAX_JUDGE_SPEND_USD=1.00)
blocked = False
try:
    ns3['_reserve_judge_call']()
except Exception as e:
    blocked = (type(e).__name__ == 'JudgeBudgetExceeded')
check('restart cannot re-arm the budget', blocked, True)

print('%d failures' % len(fails))
for f in fails:
    print('  FAIL', f)
sys.exit(1 if fails else 0)
