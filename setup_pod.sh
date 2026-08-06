#!/usr/bin/env bash
# One-shot pod bootstrap for the RLVR CoT sweep.
# The pod bills from boot, so this does everything non-interactively and fails loudly.
#
#   bash setup_pod.sh
#
# Expects .env to already be present in /workspace/rlvr-cot (scp it over first).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Aditya26189/OLMo-COT-verbalization-sweep}"
WORKDIR="${WORKDIR:-/workspace/rlvr-cot}"

echo "=============================================="
echo " 1/6  GPU sanity"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "
import torch
assert torch.cuda.is_available(), 'NO CUDA — wrong pod image or GPU not attached'
p = torch.cuda.get_device_properties(0)
print(f'GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f} GB  bf16: {torch.cuda.is_bf16_supported()}')
assert p.total_memory/1e9 > 70, f'Expected an 80GB card, got {p.total_memory/1e9:.0f}GB — wrong pod'
assert torch.cuda.is_bf16_supported(), 'No bf16 — this checkpoint is bf16-native, do not run fp16 here'
"

echo
echo "=============================================="
echo " 2/6  Disk"
echo "=============================================="
df -h /workspace | tail -1
python -c "
import shutil
free = shutil.disk_usage('/workspace').free/1e9
print(f'free: {free:.1f} GB')
assert free > 60, f'Only {free:.0f} GB free — need room for 14.6 GB checkpoints plus HF cache'
"

echo
echo "=============================================="
echo " 3/6  Repo"
echo "=============================================="
mkdir -p "$(dirname "$WORKDIR")"
if [ -d "$WORKDIR/.git" ]; then
  echo "repo already present, pulling"
  git -C "$WORKDIR" pull --ff-only || echo "(pull skipped)"
else
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git clone https://github.com/yf-he/InstrumentalEval.git --depth=1 2>/dev/null || echo "InstrumentalEval already cloned"

echo
echo "=============================================="
echo " 4/6  Dependencies (pinned)"
echo "=============================================="
pip install -q --upgrade pip
pip install -q 'transformers>=4.57.1,<5.0' 'accelerate>=0.30,<2.0' \
               'ruptures>=1.1,<2.0' 'scipy>=1.11,<2.0' 'statsmodels>=0.14,<1.0' \
               'openai>=1.30,<3.0' 'google-generativeai>=0.8,<1.0' \
               'scikit-learn>=1.3,<2.0' 'datasets>=2.18,<5.0' 'tqdm>=4.66' \
               'huggingface_hub>=0.22,<1.0' 'matplotlib>=3.8,<4.0' hf_transfer
# The RunPod image exports HF_HUB_ENABLE_HF_TRANSFER=1 but ships without the package,
# so every HuggingFace download raises. Install it; fall back to disabling the flag.
python -c 'import hf_transfer' 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
python -c "
import transformers, torch, datasets, openai
print('transformers', transformers.__version__)
print('datasets    ', datasets.__version__)
print('openai      ', openai.__version__)
assert transformers.__version__.startswith('4.'), 'transformers v5 installed — pin failed'
"

echo
echo "=============================================="
echo " 5/6  Credentials"
echo "=============================================="
if [ ! -f "$WORKDIR/.env" ]; then
  echo "*** .env NOT FOUND at $WORKDIR/.env"
  echo "*** From your laptop:  scp .env root@<POD_IP>:$WORKDIR/.env"
  exit 1
fi
chmod 600 "$WORKDIR/.env"
python -c "
from pathlib import Path
need = {'OPENAI_API_KEY', 'GEMINI_API_KEY', 'GITHUB_TOKEN'}
have = set()
for l in Path('.env').read_text(encoding='utf-8').splitlines():
    l = l.strip()
    if l and not l.startswith('#') and '=' in l and l.split('=',1)[1].strip():
        have.add(l.split('=',1)[0].strip())
print('present:', sorted(have))
missing = need - have
assert not missing, f'missing from .env: {sorted(missing)}'
"

echo
echo "=============================================="
echo " 6/6  Offline test suite (no GPU, no API calls)"
echo "=============================================="
for t in test_logic test_analysis test_budget test_pool; do
  printf '%-16s ' "$t"
  if python "$t.py" >/dev/null 2>&1; then echo PASS; else echo FAIL; fi
done

echo
echo "=============================================="
echo " READY"
echo "=============================================="
cat <<'NEXT'
==============================================
 RUN IT — one command each, in order
==============================================

  tmux new -s sweep          # so a disconnect cannot kill the run

  # 1. FREE dev smoke test. Validates the whole pipeline at zero cost.
  python run_study.py smoke

  # 2. Paid smoke test — a few cents. HARD STOP if this fails.
  JUDGE_MODE=prod python run_study.py smoke

  # 3. Phase 1 sweep (~3-4 h). Pod stops ITSELF on exit — note ';' not '&&'.
  JUDGE_MODE=prod python run_study.py phase1 2>&1 | tee phase1.log; runpodctl stop pod $RUNPOD_POD_ID

  # 4. Analysis, then Phase 2 + final results
  JUDGE_MODE=prod python run_study.py analysis 2>&1 | tee analysis.log
  JUDGE_MODE=prod python run_study.py phase2   2>&1 | tee phase2.log; runpodctl stop pod $RUNPOD_POD_ID

Detach: Ctrl+B then D.   Reattach: tmux attach -t sweep
NEXT
