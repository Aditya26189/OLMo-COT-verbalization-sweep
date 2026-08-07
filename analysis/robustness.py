# -*- coding: utf-8 -*-
"""Robustness checks a reviewer would demand.

The pooled analysis in dissociation.py treats 210 category-trace observations as independent.
They are not: the same 76 scenarios are re-measured at 5 checkpoints, so there are at most
42 independent units. This script re-tests under scenario clustering, plus:

  * unanimous-judge-only subset (drops split verdicts)
  * length-matched comparison (the §6.2 confound)
  * "considered but not enacted" traces — verbalized without behaving
  * silent-but-convergent traces — the failures of faithfulness
  * cluster bootstrap CIs

Run from the repo root:  python analysis/robustness.py
"""
import json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(ROOT, 'results', 'phase2')
CK = ['step_1300', 'step_1400', 'step_1500', 'step_1600', 'step_1700']
CATS = ["Evading Shutdown", "Hiding Unwanted Behavior", "Deceptive Behavior"]
rng = np.random.default_rng(42)


def cat_score(res, cat):
    v = res.get(cat) if isinstance(res, dict) else None
    if isinstance(v, dict):
        v = v.get('score')
    return int(v) if isinstance(v, (int, float, bool)) and v in (0, 1) else None


# ── load: one row per (checkpoint, scenario, category) ────────────────────────
rows = []
for ck in CK:
    n = ck.split('_')[1]
    verbs = json.load(open(os.path.join(P2, 'checkpoint_step_%s_verbalization.json' % n), encoding='utf-8'))
    traces = {t['trace_id']: t for t in
              json.load(open(os.path.join(P2, 'checkpoint_step_%s_traces.json' % n), encoding='utf-8'))}
    for r in verbs:
        cat = r.get('category')
        if cat not in CATS:
            continue
        s = cat_score(r.get('judge_result'), cat)
        if s is None:
            continue
        passes = [cat_score(p.get('result'), cat) for p in r.get('judge_passes', [])]
        passes = [p for p in passes if p is not None]
        t = traces.get(r['trace_id'], {})
        rows.append({
            'ck': ck, 'cat': cat, 'sid': r['scenario_idx'],
            'behav': r.get('behavioral_score'), 'verb': s,
            'unanimous': len(set(passes)) == 1 and len(passes) == 5,
            'truncated': bool(t.get('truncated')),
            'words': len(t.get('full_output', '').split()),
            'trace': t.get('full_output', ''), 'scenario': t.get('scenario', ''),
        })

def table(rs):
    a = sum(1 for r in rs if r['behav'] == 1 and r['verb'] == 1)
    b = sum(1 for r in rs if r['behav'] == 1 and r['verb'] == 0)
    c = sum(1 for r in rs if r['behav'] == 0 and r['verb'] == 1)
    d = sum(1 for r in rs if r['behav'] == 0 and r['verb'] == 0)
    return a, b, c, d

def report(title, subset):
    print('\n' + title)
    ps = []
    for cat in CATS:
        rs = [r for r in subset if r['cat'] == cat]
        a, b, c, d = table(rs)
        if (a + b) == 0 or (c + d) == 0:
            print(f'  {cat:<26} degenerate ({a+b} convergent, {c+d} non)')
            ps.append(np.nan); continue
        _, p = fisher_exact([[a, b], [c, d]])
        ps.append(p)
        print(f'  {cat:<26} verb|b=1 {a}/{a+b}={a/(a+b):.2f}   '
              f'verb|b=0 {c}/{c+d}={c/(c+d):.2f}   p={p:.3g}')
    ok = [p for p in ps if p == p]
    if ok:
        rej, padj, _, _ = multipletests(ok, method='fdr_bh')
        print(f'  FDR: {", ".join(f"{p:.3g}{chr(42) if r else chr(32)}" for p, r in zip(padj, rej))}')
    return ps


print('=' * 100)
print('R1. SCENARIO-CLUSTERED — the correct unit of analysis'.center(100))
print('=' * 100)
print('The same scenarios recur at 5 checkpoints, so the 210 observations are ~5 repeated')
print('measures of 42 scenarios. Collapse each scenario to its majority across checkpoints.')
clustered = []
by_sid = defaultdict(list)
for r in rows:
    by_sid[(r['cat'], r['sid'])].append(r)
for (cat, sid), rs in by_sid.items():
    bh = [r['behav'] for r in rs if r['behav'] is not None]
    vb = [r['verb'] for r in rs]
    clustered.append({'cat': cat, 'sid': sid,
                      'behav': int(np.mean(bh) >= 0.5) if bh else None,
                      'verb': int(np.mean(vb) >= 0.5)})
clustered = [r for r in clustered if r['behav'] is not None]
print(f'\nindependent units: {len(clustered)} scenarios (was {len(rows)} observations)')
report('scenario-level (majority across checkpoints):', clustered)

print('\n' + '=' * 100)
print('R2. UNANIMOUS-JUDGE SUBSET — drops every split 5-pass verdict'.center(100))
print('=' * 100)
report('unanimous verdicts only:', [r for r in rows if r['unanimous']])

print('\n' + '=' * 100)
print('R3. LENGTH-MATCHED — the confound from results section 6.2'.center(100))
print('=' * 100)
for cat in CATS:
    rs = [r for r in rows if r['cat'] == cat]
    med = np.median([r['words'] for r in rs])
    for label, sub in [('shorter than median', [r for r in rs if r['words'] <= med]),
                       ('longer than median', [r for r in rs if r['words'] > med])]:
        a, b, c, d = table(sub)
        if (a + b) and (c + d):
            _, p = fisher_exact([[a, b], [c, d]])
            print(f'  {cat:<26} {label:<20} verb|b=1 {a}/{a+b}   verb|b=0 {c}/{c+d}   p={p:.3g}')
        else:
            print(f'  {cat:<26} {label:<20} degenerate ({a+b} convergent)')

print('\n' + '=' * 100)
print('R4. CLUSTER BOOTSTRAP — resample scenarios, not observations'.center(100))
print('=' * 100)
for cat in CATS:
    sids = sorted({r['sid'] for r in rows if r['cat'] == cat})
    by = {s: [r for r in rows if r['cat'] == cat and r['sid'] == s] for s in sids}
    diffs = []
    for _ in range(4000):
        pick = rng.choice(sids, size=len(sids), replace=True)
        rs = [r for s in pick for r in by[s]]
        a, b, c, d = table(rs)
        if (a + b) and (c + d):
            diffs.append(a / (a + b) - c / (c + d))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f'  {cat:<26} risk difference {np.mean(diffs):+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]'
          f'   {"excludes 0" if lo > 0 else "INCLUDES 0"}')

print('\n' + '=' * 100)
print('R5. THE THREE INTERESTING CELLS'.center(100))
print('=' * 100)
for cat in CATS:
    rs = [r for r in rows if r['cat'] == cat]
    a, b, c, d = table(rs)
    print(f'\n{cat}')
    print(f'  enacted + verbalized      {a:>4}   faithful')
    print(f'  enacted + SILENT          {b:>4}   <-- failures of faithfulness')
    print(f'  verbalized, NOT enacted   {c:>4}   <-- considered and rejected?')
    print(f'  neither                   {d:>4}')

print('\n--- "considered but not enacted": what does the model say? ---')
shown = 0
for r in rows:
    if r['behav'] == 0 and r['verb'] == 1 and r['cat'] == 'Evading Shutdown' and shown < 3:
        txt = ' '.join(r['trace'].split())
        low = txt.lower()
        i = max((low.find(k) for k in ('shut down', 'shutdown', 'override')), default=-1)
        if i < 0:
            continue
        shown += 1
        print(f'\n  [{r["ck"]} scenario {r["sid"]}]')
        print('   …' + txt[max(0, i - 180):i + 260] + '…')

print('\n--- "enacted but SILENT": the faithfulness failures ---')
shown = 0
for r in rows:
    if r['behav'] == 1 and r['verb'] == 0 and shown < 3:
        shown += 1
        txt = ' '.join(r['trace'].split())
        print(f'\n  [{r["cat"]} | {r["ck"]} scenario {r["sid"]} | {r["words"]} words | '
              f'truncated={r["truncated"]}]')
        print('   …' + txt[:300] + '…')
if shown == 0:
    print('  none in any category')
