#!/bin/bash
cd "$(dirname "$0")"
source env.sh
POLL=120

echo "[daemon] watching GPU 1: waiting for ALL mdrun to finish"
while true; do
    # check if any gmx mdrun process is using GPU 1
    md_on_gpu1=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 2>/dev/null); do
        p="${p%,}"
        if ls -l /proc/$p/cwd 2>/dev/null | grep -q "/md/"; then
            sysname=$(ls -l /proc/$p/cwd 2>/dev/null | awk -F/ "{print \$NF}")
            echo "[daemon] $(date "+%H:%M") GPU 1 busy: $sysname (pid $p)"
            md_on_gpu1=1
        fi
    done
    if [ $md_on_gpu1 -eq 0 ]; then
        echo "[daemon] GPU 1 free @ $(date "+%H:%M") — starting Model B"
        export CUDA_VISIBLE_DEVICES=1
        bash train_ff.sh
        echo "[daemon] Model B finished @ $(date "+%H:%M") — starting inference"
        bash run_inference.sh
        echo "[daemon] ALL DONE @ $(date "+%H:%M")"
        break
    fi
    sleep $POLL
done
