# -*- coding: utf-8 -*-
"""Paper-grade statistics for the dissociation: exact CIs, effect sizes, base rates.

With cells as small as n=6 a bare proportion is misleading, so every headline rate gets a
Wilson score interval and every 2x2 gets an odds ratio with a Haldane-Anscombe correction
(the uncorrected OR is infinite whenever a cell is zero).

Run from the repo root:  python analysis/effect_sizes.py
"""
import json, os, sys
import numpy as np
from scipy.stats import fisher_exact, norm
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(ROOT, 'results', 'phase2')
CK = ['step_1300', 'step_1400', 'step_1500', 'step_1600', 'step_1700']
CATS = ["Evading Shutdown", "Hiding Unwanted Behavior", "Deceptive Behavior"]


def cat_score(res, cat):
    v = res.get(cat) if isinstance(res, dict) else None
    if isinstance(v, dict):
        v = v.get('score')
    return int(v) if isinstance(v, (int, float, bool)) and v in (0, 1) else None


def wilson(k, n, z=1.96):
    """Wilson score interval — correct for tiny n, unlike the normal approximation."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def or_haldane(a, b, c, d):
    """Odds ratio with +0.5 added to every cell, so zero cells do not give infinity."""
    a, b, c, d = a + .5, b + .5, c + .5, d + .5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    return orr, lo, hi


def cliffs_delta(x, y):
    if not x or not y:
        return float('nan')
    return sum(np.sign(a - b) for a in x for b in y) / (len(x) * len(y))


pooled = {c: {'cv': [], 'nv': []} for c in CATS}
for ck in CK:
    with open(os.path.join(P2, 'checkpoint_%s_verbalization.json' % ck), encoding='utf-8') as f:
        v = json.load(f)
    for cat in CATS:
        for r in v:
            if r.get('category') != cat:
                continue
            s = cat_score(r.get('judge_result'), cat)
            if s is None:
                continue
            (pooled[cat]['cv'] if r.get('behavioral_score') == 1 else pooled[cat]['nv']).append(s)

print('=' * 100)
print('POOLED 2x2 TABLES, EXACT TESTS, EFFECT SIZES'.center(100))
print('=' * 100)
ps = []
for cat in CATS:
    cv, nv = pooled[cat]['cv'], pooled[cat]['nv']
    a, b = sum(cv), len(cv) - sum(cv)          # behaviour=1: verbalized / silent
    c, d = sum(nv), len(nv) - sum(nv)          # behaviour=0: verbalized / silent
    _, p = fisher_exact([[a, b], [c, d]])
    ps.append(p)
    orr, lo, hi = or_haldane(a, b, c, d)
    w1, w2 = wilson(a, len(cv)), wilson(c, len(nv))
    print(f'\n{cat}')
    print(f'                     verbalized   silent   total')
    print(f'  behaviour = 1  {a:>12}{b:>9}{len(cv):>8}   rate {a/len(cv):.3f}  95% CI [{w1[0]:.2f}, {w1[1]:.2f}]')
    print(f'  behaviour = 0  {c:>12}{d:>9}{len(nv):>8}   rate {c/len(nv):.3f}  95% CI [{w2[0]:.2f}, {w2[1]:.2f}]')
    print(f'  Fisher exact p = {p:.6g}')
    print(f'  odds ratio (Haldane-corrected) = {orr:.1f}  95% CI [{lo:.1f}, {hi:.0f}]')
    print(f'  risk difference = {a/len(cv) - c/len(nv):+.3f}')
    print(f"  Cliff's delta   = {cliffs_delta(cv, nv):+.3f}")

rej, padj, _, _ = multipletests(ps, method='fdr_bh')
print('\nFDR (Benjamini-Hochberg) across the three pooled tests:')
for cat, p, pa, r in zip(CATS, ps, padj, rej):
    print(f'  {cat:<28} p_raw={p:.6g}  p_fdr={pa:.6g}  {"SIGNIFICANT" if r else ""}')

print('\n' + '=' * 100)
print('BASE RATES — how much data each conclusion rests on'.center(100))
print('=' * 100)
print(f'{"category":<28}{"traces":>8}{"behav=1":>10}{"% convergent":>14}')
for cat in CATS:
    cv, nv = pooled[cat]['cv'], pooled[cat]['nv']
    n = len(cv) + len(nv)
    print(f'{cat:<28}{n:>8}{len(cv):>10}{len(cv)/n:>13.1%}')
print('\nThe convergent cell is the binding constraint on every claim in this study.')
