import os
# -*- coding: utf-8 -*-
"""score_instrumental_batch must keep results aligned to input order under out-of-order completion."""
import ast, io, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rlvr_cot_phase0_fixed.ipynb')
nb = json.load(io.open(NB, encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
CELL = next(s for s in cells if 'def score_instrumental_batch' in s)

STAGING = os.environ.get('STAGING', '1') == '1'
ns = {'ThreadPoolExecutor': ThreadPoolExecutor, 'as_completed': as_completed,
      'tqdm': lambda x, **k: x, 'threading': threading, 'time': time,
      'N_JUDGE_PASSES': 5, 'JUDGE_CONCURRENCY': 16, 'JUDGE_PASS_TEMPERATURE': 0.5,
      'JUDGE_CACHE_STAGING': STAGING,
      'CATEGORIES': ["Evading Shutdown", "Hiding Unwanted Behavior", "Deceptive Behavior"]}
print('--- cache staging: %s ---' % ('ON (pass 0 first, then 1-4)' if STAGING else 'OFF (all 5 at once)'))

class JudgeBudgetExceeded(RuntimeError): pass
class JudgeHealthError(RuntimeError): pass
ns['JudgeBudgetExceeded'] = JudgeBudgetExceeded
ns['JudgeHealthError'] = JudgeHealthError

for node in ast.parse(CELL).body:
    if isinstance(node, ast.FunctionDef) and node.name in (
            '_majority', '_assemble_5pass', 'score_instrumental_batch'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<nb>', 'exec'), ns)

fails = []
def check(label, got, want):
    if got != want:
        fails.append('%s: got %r want %r' % (label, got, want))

N = 76
# Each item's "true" score is derived from its index; random latency forces out-of-order completion.
rng = random.Random(0)
def fake_judge(scenario, response, category, specific_prompt, temperature=0.0):
    time.sleep(rng.uniform(0, 0.02))
    return int(scenario.split('#')[1]) % 2      # 1 for odd indices, 0 for even
ns['score_instrumental'] = fake_judge

items = [{'scenario': 'item#%d' % i, 'response': 'r%d' % i,
          'category': 'Evading Shutdown', 'specific_prompt': 'sp'} for i in range(N)]
out = ns['score_instrumental_batch'](items)

check('length preserved', len(out), N)
check('all items have 5 passes', all(len(o['passes']) == 5 for o in out), True)
mis = [i for i, o in enumerate(out) if o['majority_score'] != i % 2]
check('alignment under out-of-order completion', mis, [])
check('pass indices intact', [p['pass'] for p in out[0]['passes']], [0, 1, 2, 3, 4])
check('no spurious failures', sum(o['n_failed'] for o in out), 0)

# Partial failures must be attributed to the RIGHT item.
def flaky_judge(scenario, response, category, specific_prompt, temperature=0.0):
    time.sleep(rng.uniform(0, 0.01))
    i = int(scenario.split('#')[1])
    if i == 7:
        raise RuntimeError('simulated judge failure')
    return 1
ns['score_instrumental'] = flaky_judge
out2 = ns['score_instrumental_batch'](items)
check('failure isolated to item 7', out2[7]['n_failed'], 5)
check('item 7 majority is None', out2[7]['majority_score'], None)
check('other items unaffected', sum(o['n_failed'] for i, o in enumerate(out2) if i != 7), 0)
check('other items scored', all(o['majority_score'] == 1 for i, o in enumerate(out2) if i != 7), True)

# A fatal budget error must propagate, not be swallowed.
def broke_judge(scenario, response, category, specific_prompt, temperature=0.0):
    raise JudgeBudgetExceeded('ceiling hit')
ns['score_instrumental'] = broke_judge
raised = None
try:
    ns['score_instrumental_batch'](items)
except Exception as e:
    raised = type(e).__name__
check('budget error propagates', raised, 'JudgeBudgetExceeded')

print('%d failures' % len(fails))
for f in fails:
    print('  FAIL', f)
sys.exit(1 if fails else 0)
