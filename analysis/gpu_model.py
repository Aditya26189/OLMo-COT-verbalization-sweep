# -*- coding: utf-8 -*-
"""Cost/throughput model for the RLVR CoT sweep on Olmo-3-7B.

Decode is memory-bandwidth bound, so throughput is modelled from bytes moved per decode step:
    bytes/step = weights + batch * seqlen * kv_bytes_per_token
    tok/s      = batch * BW / bytes_per_step * efficiency
Olmo-3-7B is MHA (num_key_value_heads == num_attention_heads == 32), so KV traffic is large
and batching saturates earlier than it would on a GQA model.
"""

# ── Model facts (from config.json on step_100) ───────────────────────────
HIDDEN, N_HEADS, N_KV_HEADS, N_LAYERS = 4096, 32, 32, 32
HEAD_DIM = HIDDEN // N_HEADS                      # 128
BYTES = 2                                          # bf16
W_BYTES = 14.596e9                                 # measured from safetensors index
KV_PER_TOK = 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * BYTES     # bytes per token
print(f'KV cache per token: {KV_PER_TOK/1024:.0f} KB  (MHA, no GQA savings)')

# ── Workload ─────────────────────────────────────────────────────────────
N_CKPT_P1, N_CKPT_P2 = 19, 5
N_SCEN, N_MATH = 76, 100
TOK_INST, TOK_MATH, TOK_P2 = 1000, 1536, 1536
SEQ_PEAK = 2500                                    # prompt + generation, worst case

p1_tokens = N_CKPT_P1 * (N_SCEN * TOK_INST + N_MATH * TOK_MATH)
p2_tokens = N_CKPT_P2 * (N_SCEN * TOK_P2)
TOTAL_TOK = p1_tokens + p2_tokens
print(f'Generation budget: {p1_tokens/1e6:.2f}M (P1) + {p2_tokens/1e6:.2f}M (P2) '
      f'= {TOTAL_TOK/1e6:.2f}M tokens (upper bound: assumes no early EOS)')

# ── Fixed overhead that does NOT shrink on a faster GPU ──────────────────
N_LOADS = N_CKPT_P1 + N_CKPT_P2                    # 24 checkpoint loads
DOWNLOAD_H = N_LOADS * 14.6 / 0.30 / 3600          # 14.6 GB at ~300 MB/s
LOAD_H     = N_LOADS * 75 / 3600                   # ~75 s shard load + placement
N_PASSES, JUDGE_CONC = 5, 16
# Judging runs after generation with the GPU idle but billing. With the pooled judge phase
# all N_SCEN*N_PASSES calls per checkpoint go out at JUDGE_CONC concurrency.
JUDGE_IDLE_H = (N_CKPT_P1 * N_SCEN * N_PASSES * 4.0 / JUDGE_CONC
                + N_CKPT_P2 * N_SCEN * N_PASSES * 8.0 / JUDGE_CONC) / 3600
_SERIAL_IDLE_H = (N_CKPT_P1 * N_SCEN * 4.0 + N_CKPT_P2 * N_SCEN * 8.0) / 3600
print(f'Judge idle: {JUDGE_IDLE_H:.2f}h pooled (was {_SERIAL_IDLE_H:.2f}h serial)')
OVERHEAD_H = DOWNLOAD_H + LOAD_H + JUDGE_IDLE_H
print(f'Fixed overhead: download {DOWNLOAD_H:.2f}h + load {LOAD_H:.2f}h + '
      f'judge-idle {JUDGE_IDLE_H:.2f}h = {OVERHEAD_H:.2f}h (GPU billed, not generating)\n')

EFF = 0.25          # HF transformers eager-attention efficiency vs theoretical bandwidth
VRAM_OVERHEAD_GB = 2.0

# name: (VRAM GB, BW GB/s, secure $/hr, community $/hr, availability, secure_ok)
GPUS = {
    'RTX A5000':      (24,  768, 0.27, 0.16, 'LOW',    True),
    'RTX 3090':       (24,  936, 0.50, 0.22, 'LOW',    True),
    'RTX 4090':       (24, 1008, 0.74, 0.34, 'HIGH',   True),
    'L4':             (24,  300, 0.49, 0.44, 'MEDIUM', True),
    'RTX 5090':       (32, 1792, 0.99, 0.69, 'HIGH',   True),
    'A40':            (48,  696, 0.44, 0.35, 'MEDIUM', True),
    'RTX A6000':      (48,  768, 0.53, 0.33, 'LOW',    True),
    'L40S':           (48,  864, 0.99, 0.79, 'LOW',    True),
    'RTX 6000 Ada':   (48,  960, 0.84, 0.74, 'LOW',    True),
    'A100 PCIe 80':   (80, 1935, 1.39, 1.19, 'LOW',    True),
    'A100 SXM 80':    (80, 2039, 1.59, 1.39, 'HIGH',   True),
    'RTX PRO 6000':   (96, 1792, 2.09, 1.69, 'HIGH',   True),
    'H100 PCIe':      (80, 2000, 2.89, 1.99, 'LOW',    True),
    'H100 SXM':       (80, 3350, 3.29, 2.69, 'HIGH',   True),
    'H200 SXM':      (141, 4800, 4.59, 3.59, 'MEDIUM', True),
    'B200':          (180, 8000, 6.79, 5.98, 'LOW',    True),
}

def max_batch(vram_gb):
    """Largest power-of-two batch fitting weights + peak KV cache + overhead."""
    budget = vram_gb - W_BYTES / 1e9 - VRAM_OVERHEAD_GB
    per_seq = SEQ_PEAK * KV_PER_TOK / 1e9
    b = int(budget / per_seq)
    out = 1
    while out * 2 <= b:
        out *= 2
    return out, budget / per_seq

def throughput(bw_gbs, batch, seq_avg=1250):
    bytes_step = W_BYTES + batch * seq_avg * KV_PER_TOK
    return batch * (bw_gbs * 1e9) / bytes_step * EFF

rows = []
for name, (vram, bw, sec, comm, avail, sec_ok) in GPUS.items():
    b, braw = max_batch(vram)
    tps = throughput(bw, b)
    gen_h = TOTAL_TOK / tps / 3600
    total_h = gen_h + OVERHEAD_H
    rows.append((name, vram, bw, b, braw, tps, gen_h, total_h,
                 total_h * sec, total_h * comm, avail))

rows.sort(key=lambda r: r[8])

print(f'{"GPU":<15}{"VRAM":>5}{"BW":>6}{"batch":>6}{"tok/s":>7}{"gen_h":>7}'
      f'{"tot_h":>7}{"$secure":>9}{"$comm":>8}  avail')
print('-' * 82)
for (name, vram, bw, b, braw, tps, gen_h, tot_h, csec, ccom, avail) in rows:
    print(f'{name:<15}{vram:>5}{bw:>6}{b:>6}{tps:>7.0f}{gen_h:>7.1f}'
          f'{tot_h:>7.1f}{csec:>9.2f}{ccom:>8.2f}  {avail}')

print(f'\n(efficiency factor {EFF:.0%} of theoretical bandwidth; seq_avg 1250 tok; '
      f'{OVERHEAD_H:.1f}h fixed overhead added to every option)')

# ── Sensitivity on the efficiency assumption ─────────────────────────────
print('\nSensitivity — total $ (secure) if HF efficiency is 20% / 25% / 35%:')
for name in ['RTX 5090', 'RTX A6000', 'A40', 'A100 SXM 80', 'RTX PRO 6000', 'H100 SXM', 'H200 SXM']:
    vram, bw, sec, comm, avail, _ = GPUS[name]
    b, _ = max_batch(vram)
    cells = []
    for e in (0.20, 0.25, 0.35):
        tps = batch_tps = b * (bw * 1e9) / (W_BYTES + b * 1250 * KV_PER_TOK) * e
        cells.append((TOTAL_TOK / tps / 3600 + OVERHEAD_H) * sec)
    print(f'  {name:<15} ${cells[0]:6.2f} / ${cells[1]:6.2f} / ${cells[2]:6.2f}')

# ── What if the judge idle is removed? ───────────────────────────────────
print(f'\nIf judge-idle ({JUDGE_IDLE_H:.1f}h) is overlapped/parallelised away:')
for name in ['A40', 'RTX A6000', 'A100 SXM 80', 'H100 SXM']:
    vram, bw, sec, comm, avail, _ = GPUS[name]
    b, _ = max_batch(vram)
    tps = throughput(bw, b)
    h_now = TOTAL_TOK / tps / 3600 + OVERHEAD_H
    h_opt = TOTAL_TOK / tps / 3600 + (OVERHEAD_H - JUDGE_IDLE_H)
    print(f'  {name:<15} {h_now:5.1f}h -> {h_opt:5.1f}h   ${h_now*sec:6.2f} -> ${h_opt*sec:6.2f}'
          f'   (saves ${(h_now-h_opt)*sec:.2f})')
