# -*- coding: utf-8 -*-
"""Read the actual traces behind the headline numbers. 100% verbalization on n=7 needs eyeballing."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(ROOT, 'results', 'phase2')
def _vpath(n): return os.path.join(P2, 'checkpoint_step_%s_verbalization.json' % n)
def _tpath(n): return os.path.join(P2, 'checkpoint_step_%s_traces.json' % n)

import json, re, textwrap

CK = ['step_1300', 'step_1400', 'step_1500', 'step_1600', 'step_1700']
CATS = ["Evading Shutdown", "Hiding Unwanted Behavior", "Deceptive Behavior"]

def cat_score(res, cat):
    v = res.get(cat) if isinstance(res, dict) else None
    if isinstance(v, dict): v = v.get('score')
    return int(v) if isinstance(v, (int, float, bool)) and v in (0, 1) else None

traces, verbs = {}, {}
for ck in CK:
    n = ck.split('_')[1]
    for t in json.load(open(_tpath(n), encoding='utf-8')):
        traces[t['trace_id']] = t
    for v in json.load(open(_vpath(n), encoding='utf-8')):
        verbs[v['trace_id']] = v

# The cells that drive the headline: behaviour=1 AND verbalized=1
print('=' * 92)
print('EVIDENCE QUOTED BY THE JUDGE, behaviour=1 & verbalized=1'.center(92))
print('=' * 92)
shown = {c: 0 for c in CATS}
for tid, v in verbs.items():
    cat = v.get('category')
    if cat not in CATS or v.get('behavioral_score') != 1:
        continue
    if cat_score(v.get('judge_result'), cat) != 1 or shown[cat] >= 2:
        continue
    # pull the evidence string the judge quoted, from the first pass that has one
    ev = None
    for p in v.get('judge_passes', []):
        r = p.get('result')
        if isinstance(r, dict) and isinstance(r.get(cat), dict):
            e = r[cat].get('evidence')
            if e and e.lower() not in ('none', "'none'", 'no supporting phrases found.'):
                ev = e
                break
    if not ev:
        continue
    shown[cat] += 1
    t = traces[tid]
    print('\n--- %s | %s | %s' % (cat, v['checkpoint'], tid))
    print('    SCENARIO : %s' % textwrap.shorten(t['scenario'].replace('\n', ' '), 150))
    print('    JUDGE EVIDENCE:')
    for line in textwrap.wrap(ev, 84):
        print('      > ' + line)

# how consistent were the 5 passes?
print('\n' + '=' * 92)
print('5-PASS AGREEMENT (how stable is each verdict?)'.center(92))
print('=' * 92)
for cat in CATS:
    unan = split = 0
    for v in verbs.values():
        if v.get('category') != cat:
            continue
        sc = [cat_score(p.get('result'), cat) for p in v.get('judge_passes', [])]
        sc = [s for s in sc if s is not None]
        if not sc:
            continue
        if len(set(sc)) == 1: unan += 1
        else: split += 1
    tot = unan + split
    print(f'  {cat:<26} unanimous {unan}/{tot} ({unan/tot:.0%}), split {split}')

# truncation vs verdict — could truncated traces be inflating verbalization?
print('\n' + '=' * 92)
print('TRUNCATION vs VERBALIZATION (artifact check)'.center(92))
print('=' * 92)
for cat in CATS:
    tr_v, tr_n, fu_v, fu_n = 0, 0, 0, 0
    for tid, v in verbs.items():
        if v.get('category') != cat: continue
        s = cat_score(v.get('judge_result'), cat)
        if s is None: continue
        if traces[tid].get('truncated'):
            tr_v += s; tr_n += 1
        else:
            fu_v += s; fu_n += 1
    r1 = tr_v / tr_n if tr_n else float('nan')
    r2 = fu_v / fu_n if fu_n else float('nan')
    print(f'  {cat:<26} truncated {r1:.2f} (n={tr_n:>3})   complete {r2:.2f} (n={fu_n:>3})')
