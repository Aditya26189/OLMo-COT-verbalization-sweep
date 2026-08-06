#!/usr/bin/env python
"""Headless runner for the RLVR CoT study. No Jupyter needed.

    python run_study.py smoke      # dev-mode smoke test (FREE)
    JUDGE_MODE=prod python run_study.py smoke     # prod smoke test (cents)
    JUDGE_MODE=prod python run_study.py phase1    # Phase 1 sweep
    JUDGE_MODE=prod python run_study.py analysis  # Phase 1 analysis + plots
    JUDGE_MODE=prod python run_study.py phase2    # lock prompt + Phase 2 + final results

Executes the notebook's own cells in order, so there is exactly one copy of the logic.
JUDGE_MODE from the environment overrides the notebook default.
"""
import io, json, os, sys, time

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rlvr_cot_phase0_fixed.ipynb')

# Ordered cell selectors. Each stage runs every cell listed for it, in this order.
SELECTORS = {
    'setup': [
        'import ruptures as rpt',            # imports
        'JUDGE_MODE = "dev"',                # config
        'def enumerate_checkpoints',         # checkpoints
        'def load_instrumental_eval',        # benchmark + judge
        'def load_math_subset',              # MATH
        'def generate_batch',                # inference utils
        'GIT_REMOTE = ',                     # git
    ],
    'smoke':    ['=== SMOKE TEST ==='],
    'phase1':   ['def sweep_checkpoint_phase1'],
    'analysis': ['def load_phase1_results', 'PELT penalty (BIC)', 'PHASE 1 PLOTS'],
    'phase2':   ['JUDGE_PROMPT = ', 'def sample_for_human_annotation',
                 'def sweep_checkpoint_phase2', 'VERBALIZATION RATES ACROSS',
                 'VERBALIZATION-MATH SPEARMAN', 'SAVE COMPLETE RESULTS SUMMARY'],
}
STAGES = {
    'smoke':    SELECTORS['setup'] + SELECTORS['smoke'],
    'phase1':   SELECTORS['setup'] + SELECTORS['phase1'],
    'analysis': SELECTORS['setup'] + SELECTORS['analysis'],
    'phase2':   SELECTORS['setup'] + SELECTORS['analysis'] + SELECTORS['phase2'],
}


def main():
    stage = (sys.argv[1] if len(sys.argv) > 1 else 'smoke').lower()
    if stage not in STAGES:
        print('usage: python run_study.py [%s]' % '|'.join(STAGES))
        return 2

    nb = json.load(io.open(NB, encoding='utf-8'))
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

    def find(marker):
        for s in code:
            if marker in s:
                return s
        raise SystemExit('FATAL: no notebook cell matching %r — notebook structure changed' % marker)

    mode = os.environ.get('JUDGE_MODE', '').strip().lower()
    if mode and mode not in ('dev', 'prod'):
        raise SystemExit('JUDGE_MODE must be dev or prod, got %r' % mode)

    print('=' * 70)
    print(' stage: %s   JUDGE_MODE: %s' % (stage, mode or 'dev (notebook default)'))
    print('=' * 70, flush=True)

    ns = {'__name__': '__main__'}
    t0 = time.time()
    for marker in STAGES[stage]:
        src = find(marker)
        # strip notebook magics / shell escapes — setup_pod.sh already installed everything
        src = ''.join('pass\n' if l.lstrip().startswith(('!', '%')) else l
                      for l in src.splitlines(keepends=True))
        if mode:
            src = src.replace('JUDGE_MODE = "dev"', 'JUDGE_MODE = "%s"' % mode)
        label = marker[:44]
        print('\n' + '-' * 70)
        print('>>> %s' % label, flush=True)
        print('-' * 70, flush=True)
        try:
            exec(compile(src, '<%s>' % label, 'exec'), ns)
        except Exception as e:
            print('\n' + '!' * 70)
            print('FAILED in cell %r: %s: %s' % (label, type(e).__name__, e))
            print('!' * 70, flush=True)
            import traceback
            traceback.print_exc()
            # Surface spend so far even on failure — this is the number that matters.
            if 'judge_stats' in ns:
                print('\njudge usage at failure:', ns['judge_stats']())
            if 'gpu_budget_status' in ns:
                print('gpu budget at failure:', ns['gpu_budget_status']())
            return 1

    print('\n' + '=' * 70)
    print(' STAGE %s COMPLETE in %.1f min' % (stage.upper(), (time.time() - t0) / 60))
    if 'judge_stats' in ns:
        print(' judge usage :', ns['judge_stats']())
    if 'gpu_budget_status' in ns:
        print(' gpu budget  :', ns['gpu_budget_status']())
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
