#!/usr/bin/env bash
# bioemu_sample_all.sh — 串行采样全部 7 个 benchmark（后台长任务）。
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
N_GEN="${N_GEN:-4000}"
CKPT="${CKPT:-../results/checkpoints/flow_model_r15.pt}"
LOG=../results/ago2/bioemu_bench/sampling_all.log

for b in ood60 oodval domainmotion localunfolding crypticpocket folding_free_energies md_emulation; do
  echo "===== $(date '+%F %T') 开始 $b =====" >> "$LOG"
  $PY dynaflow_bench_adapter.py --sample --benchmark "$b" --n-gen "$N_GEN" --ckpt "$CKPT" >> "$LOG" 2>&1
  echo "===== $(date '+%F %T') 完成 $b =====" >> "$LOG"
done
echo "===== $(date '+%F %T') 全部 benchmark 采样完成 =====" >> "$LOG"
