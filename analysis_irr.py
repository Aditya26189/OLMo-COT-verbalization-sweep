# -*- coding: utf-8 -*-
"""Inter-rater reliability: GPT-4o (primary) vs GPT-4o-mini (second judge).

~60 paid calls, ~$0.30. Single-judge results are not defensible for publication.
Uses the pre-registered judge_prompt.txt verbatim.
"""
import json, os, random, re, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from sklearn.metrics import cohen_kappa_score

CATS = ["Evading Shutdown", "Hiding Unwanted Behavior", "Deceptive Behavior"]
N_SAMPLE = 30
ENV = Path(r'c:\Users\LawLight\Desktop\rlvr-cot\.env')
key = next((l.split('=', 1)[1].strip() for l in ENV.read_text(encoding='utf-8').splitlines()
            if not l.strip().startswith('#') and l.strip().startswith('OPENAI_API_KEY=')), None)
assert key, 'no OPENAI_API_KEY in .env'
client = openai.OpenAI(api_key=key)

PROMPT = urllib.request.urlopen(
    'https://raw.githubusercontent.com/Aditya26189/OLMo-COT-verbalization-sweep/master/judge_prompt.txt'
).read().decode('utf-8')
assert '{trace}' in PROMPT
print('pre-registered judge prompt loaded (%d chars)' % len(PROMPT))


def parse_json(raw):
    if not raw: return None
    try:
        o = json.loads(raw)
        if isinstance(o, dict): return o
    except Exception: pass
    best = None
    for st in (m.start() for m in re.finditer(r'\{', raw)):
        d, ins, esc = 0, False, False
        for i in range(st, len(raw)):
            ch = raw[i]
            if ins:
                if esc: esc = False
                elif ch == '\\': esc = True
                elif ch == '"': ins = False
                continue
            if ch == '"': ins = True
            elif ch == '{': d += 1
            elif ch == '}':
                d -= 1
                if d == 0:
                    try: o = json.loads(raw[st:i + 1])
                    except Exception: break
                    if isinstance(o, dict) and any(c in o for c in CATS): best = o
                    break
    return best

def sc(res, cat):
    v = res.get(cat) if isinstance(res, dict) else None
    if isinstance(v, dict): v = v.get('score')
    return int(v) if isinstance(v, (int, float, bool)) and v in (0, 1) else None

COST = {'in': 0, 'out': 0, 'in_mini': 0, 'out_mini': 0}
def judge(trace, model):
    r = client.chat.completions.create(
        model=model, messages=[{'role': 'user', 'content': PROMPT.replace('{trace}', trace)}],
        max_tokens=1500, temperature=0.0)
    u = r.usage
    if 'mini' in model:
        COST['in_mini'] += u.prompt_tokens; COST['out_mini'] += u.completion_tokens
    else:
        COST['in'] += u.prompt_tokens; COST['out'] += u.completion_tokens
    return parse_json(r.choices[0].message.content)

# sample traces, stratified by category
pool = []
for n in ['1300', '1400', '1500', '1600', '1700']:
    for t in json.load(open('t%s.json' % n, encoding='utf-8')):
        if t.get('category') in CATS:
            pool.append(t)
rng = random.Random(42)
per = N_SAMPLE // len(CATS)
sample = []
for c in CATS:
    sub = [t for t in pool if t['category'] == c]
    sample += rng.sample(sub, min(per, len(sub)))
print('sampled %d traces (%d per category) from %d eligible' % (len(sample), per, len(pool)))

res = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {}
    for t in sample:
        futs[ex.submit(judge, t['full_output'], 'gpt-4o')] = (t['trace_id'], 'primary')
        futs[ex.submit(judge, t['full_output'], 'gpt-4o-mini')] = (t['trace_id'], 'second')
    for f in as_completed(futs):
        tid, which = futs[f]
        try: res.setdefault(tid, {})[which] = f.result()
        except Exception as e:
            print('  call failed (%s/%s): %s' % (tid, which, e))
            res.setdefault(tid, {})[which] = None

print('\n' + '=' * 80)
print('INTER-JUDGE AGREEMENT  gpt-4o vs gpt-4o-mini'.center(80))
print('=' * 80)
print(f'{"category":<28}{"agreement":>11}{"kappa":>9}{"n":>6}')
allp, alls = [], []
for cat in CATS:
    p, s = [], []
    for t in sample:
        r = res.get(t['trace_id'], {})
        a, b = sc(r.get('primary'), cat), sc(r.get('second'), cat)
        if a is not None and b is not None:
            p.append(a); s.append(b)
    if len(p) < 5:
        print(f'{cat:<28}{"insufficient":>11}{"":>9}{len(p):>6}'); continue
    agree = sum(x == y for x, y in zip(p, s)) / len(p)
    k = cohen_kappa_score(p, s) if len(set(p + s)) > 1 else 1.0
    allp += p; alls += s
    print(f'{cat:<28}{agree:>10.1%}{k:>9.3f}{len(p):>6}')
if allp:
    agree = sum(x == y for x, y in zip(allp, alls)) / len(allp)
    k = cohen_kappa_score(allp, alls) if len(set(allp + alls)) > 1 else 1.0
    print(f'{"POOLED":<28}{agree:>10.1%}{k:>9.3f}{len(allp):>6}')
    print()
    print('  kappa > 0.8 strong | > 0.6 acceptable | < 0.6 needs justification')

cost = (COST['in'] * 2.50 + COST['out'] * 10.00) / 1e6 + \
       (COST['in_mini'] * 0.15 + COST['out_mini'] * 0.60) / 1e6
print('\ncost of this IRR run: $%.3f  (%d gpt-4o calls + %d mini calls)'
      % (cost, len(sample), len(sample)))
json.dump({'sample': [t['trace_id'] for t in sample], 'results': res,
           'cost_usd': round(cost, 4)},
          open('irr_results.json', 'w', encoding='utf-8'), indent=2)
print('saved irr_results.json')
