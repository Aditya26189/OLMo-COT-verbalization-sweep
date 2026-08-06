# -*- coding: utf-8 -*-
"""Integration test: run the notebook's Phase 1 analysis cells against synthetic results."""
import ast, io, json, os, re, sys, shutil, tempfile, contextlib

import numpy as np
import ruptures as rpt
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

NB = r'c:\Users\LawLight\Desktop\rlvr-cot\rlvr_cot_phase0_fixed.ipynb'
nb = json.load(io.open(NB, encoding='utf-8'))
code_cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
by_idx = dict(code_cells)

# The two analysis cells: load_phase1_results/series, then the statistics.
LOAD_CELL = next(i for i, s in code_cells if 'def load_phase1_results' in s)
STAT_CELL = next(i for i, s in code_cells if 'PELT penalty (BIC)' in s)
print('analysis cells: load=%d stats=%d' % (LOAD_CELL, STAT_CELL))

CATS = ["Evading Shutdown", "Hiding Unwanted Behavior", "Deceptive Behavior"]
CAT_N = {"Evading Shutdown": 20, "Hiding Unwanted Behavior": 11, "Deceptive Behavior": 11}
OTHER = {"Alignment Drift - Hacking Computer Systems": 10,
         "Alignment Drift - Running Multiple AI Copies": 13,
         "Alignment Drift - Strategically Appearing Aligned": 11}


def write_synthetic(root, ckpts, inst_rates, math_accs):
    """Write phase1 JSON files whose per-scenario scores realise the given rates."""
    Path(root, 'results/phase1').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for ckpt, rate, acc in zip(ckpts, inst_rates, math_accs):
        rows, idx = [], 0
        for cat, n in CAT_N.items():
            k = int(round(rate * n))
            for j in range(n):
                rows.append({'scenario_idx': idx, 'raw_category': 'Alignment Drift - ' + cat,
                             'category': cat, 'response': 'x ' * 50,
                             'score': 1 if j < k else 0,
                             'passes': [{'pass': p, 'score': 1 if j < k else 0} for p in range(5)],
                             'n_failed': 0, 'truncated': False,
                             'response_length': int(50 + 5 * rng.standard_normal())})
                idx += 1
        for stem, n in OTHER.items():
            k = int(round(rate * n))
            for j in range(n):
                rows.append({'scenario_idx': idx, 'raw_category': stem, 'category': None,
                             'response': 'x ' * 50, 'score': 1 if j < k else 0,
                             'passes': [{'pass': p, 'score': 1 if j < k else 0} for p in range(5)],
                             'n_failed': 0, 'truncated': False, 'response_length': 50})
                idx += 1
        Path(root, 'results/phase1/checkpoint_%s_instrumental.json' % ckpt).write_text(
            json.dumps(rows), encoding='utf-8')
        m = int(round(acc * 100))
        Path(root, 'results/phase1/checkpoint_%s_math.json' % ckpt).write_text(
            json.dumps([{'math_idx': i, 'score': 1 if i < m else 0, 'truncated': False}
                        for i in range(100)]), encoding='utf-8')


def run_case(name, inst_rates, math_accs, expect_method=None, expect_anchor_near=None):
    ckpts = ['step_%d' % (100 * (i + 1)) for i in range(len(inst_rates))]
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        write_synthetic(tmp, ckpts, inst_rates, math_accs)
        os.chdir(tmp)
        g = {'np': np, 'os': os, 'json': json, 're': re, 'Path': Path, 'rpt': rpt,
             'stats': stats, 'sm': sm,
             'ALL_CHECKPOINTS': ckpts, 'CATEGORIES': CATS,
             '_safe_name': lambda s: re.sub(r'[^A-Za-z0-9._-]', '_', str(s)),
             'HE_ET_AL_HIDING_BASELINE': 0.5637, 'HE_ET_AL_ARXIV': 'arXiv:2502.12206',
             'BASELINES_VERIFIED': False}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(by_idx[LOAD_CELL], '<load>', 'exec'), g)
            exec(compile(by_idx[STAT_CELL], '<stat>', 'exec'), g)
        out = buf.getvalue()
        saved = json.loads(Path('results/inflection_point.json').read_text(encoding='utf-8'))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    ok = True
    if expect_method and saved['fallback_method'] != expect_method:
        ok = False
        print('  FAIL method: got %s want %s' % (saved['fallback_method'], expect_method))
    if expect_anchor_near is not None:
        if saved['inflection_idx'] is None or abs(saved['inflection_idx'] - expect_anchor_near) > 2:
            ok = False
            print('  FAIL anchor: got %s want ~%s' % (saved['inflection_idx'], expect_anchor_near))
    if saved.get('analyzed_steps') != ckpts:
        ok = False
        print('  FAIL analyzed_steps not persisted correctly')
    print('%-28s %s  method=%-18s idx=%-5s step=%s' % (
        name, 'OK ' if ok else 'FAIL', saved['fallback_method'],
        saved['inflection_idx'], saved['inflection_step']))
    return ok, saved, out


results = []

# 1. Sharp step change at index 12 of 19 — the scenario the study is designed to detect.
rates = [0.10] * 12 + [0.55] * 7
math = list(np.linspace(0.10, 0.55, 19))
results.append(run_case('step change @12', rates, math, 'pelt_single', 12))

# 2. Same, but at index 7 — NOT a multiple of 5. The old jump=5 default could not place it here.
rates2 = [0.12] * 7 + [0.60] * 12
results.append(run_case('step change @7 (non-mult-5)', rates2, math, 'pelt_single', 7))

# 3. Gradual ramp, no discrete changepoint -> max-gradient fallback must fire.
rates3 = list(np.linspace(0.10, 0.20, 19))
results.append(run_case('gentle ramp', rates3, math, None, None))

# 4. Perfectly flat -> genuine kill condition.
rates4 = [0.20] * 19
ok4, saved4, _ = run_case('flat series', rates4, math, 'max_gradient', None)
if not saved4['kill_condition']:
    print('  FAIL: flat series should set kill_condition=True')
    ok4 = False
results.append((ok4, saved4, ''))

# 5. Prove the jump=5 default would have missed case 2.
sig = np.array(rates2, dtype=float).reshape(-1, 1)
pen = float(np.log(len(rates2)) * np.var(rates2))
cp_j1 = rpt.Pelt(model='l2', min_size=2, jump=1).fit(sig).predict(pen=pen)
cp_j5 = rpt.Pelt(model='l2', min_size=2, jump=5).fit(sig).predict(pen=pen)
print('\njump=1 changepoints: %s   (true break at 7)' % cp_j1)
print('jump=5 changepoints: %s   <- ruptures default, cannot land on 7' % cp_j5)
regression_fixed = (7 in cp_j1) and (7 not in cp_j5)
print('jump fix demonstrably matters: %s' % regression_fixed)

allok = all(r[0] for r in results) and regression_fixed
print('\n%s' % ('ALL ANALYSIS TESTS PASSED' if allok else 'SOME TESTS FAILED'))
sys.exit(0 if allok else 1)
