#!/usr/bin/env bash
# benchmark_all.sh — evaluate every incremental checkpoint against the systems
# NOT in its training pool (the held-out set). Checkpoint r{K} was trained on the
# first K systems in incr_order.txt, so its held-out = systems K..end.
#
# Only held-out systems that have a featurized npz are benchmarked. Produces
# results/bench_flow_model_r{K}.csv per checkpoint.
#
# Usage:  N_GEN=50 bash benchmark_all.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
N_GEN="${N_GEN:-50}"
CKPT_DIR="$RESULTS_DIR/checkpoints"
ORDER_FILE="$RESULTS_DIR/incr_order.txt"
[ -s "$ORDER_FILE" ] || { echo "[bench] no $ORDER_FILE — run incremental_loop.sh first"; exit 1; }
mapfile -t ORDER < <(grep -v '^$' "$ORDER_FILE")
N=${#ORDER[@]}
echo "[bench] $N systems in order: ${ORDER[*]}"

shopt -s nullglob
for ckpt in "$CKPT_DIR"/flow_model_r*.pt; do
    [ -s "$ckpt" ] || continue
    base=$(basename "$ckpt" .pt)       # e.g. flow_model_r15
    K=${base##*_r}                      # 3
    # held-out = ORDER[K..N-1] that have an npz
    HELD=""
    for ((i=K; i<N; i++)); do
        pid="${ORDER[$i]}"
        [ -s "$DATA_DIR/md/$pid.npz" ] && HELD="$HELD,$pid"
    done
    HELD="${HELD#,}"
    [ -n "$HELD" ] || { echo "[bench] $base (K=$K): nothing held out + featurized, skip"; continue; }
    IFS=,; TRAINED="${ORDER[*]:0:K}"; unset IFS
    OUT="$RESULTS_DIR/bench_${base}.csv"
    echo "[bench] $base: trained=[$TRAINED] held-out=[$HELD]"
    python3 "$PIPELINE_DIR/07_benchmark.py" --ckpt "$ckpt" --systems "$HELD" \
        --n-gen "$N_GEN" --out "$OUT" \
        || echo "[bench] FAIL $base"
done
echo "[bench] DONE — per-checkpoint CSVs in $RESULTS_DIR/bench_*.csv"
echo "[bench] headline summary: bench_${base}.csv (last ckpt)"
