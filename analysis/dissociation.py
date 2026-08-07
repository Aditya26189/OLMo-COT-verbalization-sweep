# -*- coding: utf-8 -*-
"""Behaviour-vs-verbalization dissociation on the real Phase 2 results."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(ROOT, 'results', 'phase2')
def _vpath(n): return os.path.join(P2, 'checkpoint_step_%s_verbalization.json' % n)
def _tpath(n): return os.path.join(P2, 'checkpoint_step_%s_traces.json' % n)

import json, os
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

CK = ['step_1300', 'step_1400', 'step_1500', 'step_1600', 'step_1700']
CATS = ["Evading Shutdown", "Hiding Unwanted Behavior", "Deceptive Behavior"]
CHEN = 0.02

def score(res, cat):
    v = res.get(cat) if isinstance(res, dict) else None
    if isinstance(v, dict): v = v.get('score')
    return int(v) if isinstance(v, (int, float, bool)) and v in (0, 1) else None

rows, pooled = [], {c: {'cv': [], 'nv': []} for c in CATS}
nfail = 0
print('=' * 96)
print('VERBALIZATION GIVEN BEHAVIOUR'.center(96))
print('=' * 96)
print(f'{"checkpoint":>11} {"category":<26} {"behav=1":>16} {"behav=0":>16} {"fisher p":>10} {"OR":>7}')
for ck in CK:
    v = json.load(open(_vpath(ck.split('_')[1]), encoding='utf-8'))
    nfail += sum(1 for r in v if r.get('n_failed', 0) == 5)
    for cat in CATS:
        rs = [(r, score(r.get('judge_result'), cat)) for r in v if r.get('category') == cat]
        rs = [(r, s) for r, s in rs if s is not None]
        cv = [s for r, s in rs if r.get('behavioral_score') == 1]
        nv = [s for r, s in rs if r.get('behavioral_score') == 0]
        pooled[cat]['cv'] += cv
        pooled[cat]['nv'] += nv
        if cv and nv:
            t = [[sum(cv), len(cv) - sum(cv)], [sum(nv), len(nv) - sum(nv)]]
            orr, p = fisher_exact(t)
            rows.append((ck, cat, p))
        else:
            orr, p = float('nan'), float('nan')
        f = lambda a: (f'{np.mean(a):.2f} (n={len(a)})' if a else '     — (n=0)')
        print(f'{ck:>11} {cat:<26} {f(cv):>16} {f(nv):>16} {p:>10.4f} {orr:>7.2f}')
print(f'\ntraces with all 5 judge passes failed: {nfail}')

print('\n' + '=' * 96)
print('FISHER TESTS, FDR-CORRECTED ACROSS ALL 14'.center(96))
print('=' * 96)
ps = [p for _, _, p in rows]
rej, padj, _, _ = multipletests(ps, method='fdr_bh')
for (ck, cat, p), pa, r in sorted(zip(rows, padj, rej), key=lambda x: x[0][2]):
    print(f'  {ck}/{cat:<26} p_raw={p:.4f}  p_fdr={pa:.4f}  {"SIGNIFICANT" if r else ""}')
print(f'\n  survive FDR at 0.05: {sum(rej)} of {len(rows)}')

print('\n' + '=' * 96)
print('POOLED ACROSS ALL 5 CHECKPOINTS'.center(96))
print('=' * 96)
print(f'{"category":<26}{"verb|behav=1":>16}{"verb|behav=0":>16}{"fisher p":>11}{"OR":>8}')
pooled_p = []
for cat in CATS:
    cv, nv = pooled[cat]['cv'], pooled[cat]['nv']
    t = [[sum(cv), len(cv) - sum(cv)], [sum(nv), len(nv) - sum(nv)]]
    orr, p = fisher_exact(t)
    pooled_p.append(p)
    print(f'{cat:<26}{np.mean(cv):>10.3f} (n={len(cv):>3}){np.mean(nv):>10.3f} (n={len(nv):>3})'
          f'{p:>11.5f}{orr:>8.2f}')
rej2, padj2, _, _ = multipletests(pooled_p, method='fdr_bh')
print('\n  FDR-corrected: ' + ', '.join(
    f'{c.split()[0]} p={pa:.4f}{"*" if r else ""}' for c, pa, r in zip(CATS, padj2, rej2)))

print('\n' + '=' * 96)
print('OVERALL VERBALIZATION vs CHEN ET AL. CEILING (<2% in 5/6 RL envs)'.center(96))
print('=' * 96)
for cat in CATS:
    allv = pooled[cat]['cv'] + pooled[cat]['nv']
    rate = np.mean(allv)
    print(f'  {cat:<26} {rate:>6.1%}  ({sum(allv)}/{len(allv)})   '
          f'{rate / CHEN:>5.0f}x the 2% ceiling')
