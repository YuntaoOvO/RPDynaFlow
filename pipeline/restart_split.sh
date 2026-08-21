#!/bin/bash
cd "$(dirname "$0")/.."

# Kill everything first
for pid in $(pgrep -f "incremental_loop|flow_model|gmx mdrun"); do
    kill $pid 2>/dev/null
done
sleep 2

# GPU 0 = MD only, GPU 1 = training only
cd pipeline
source env.sh

# Start ablation on GPU 1
CUDA_VISIBLE_DEVICES=1 EPOCHS_PER_ROUND=150 BENCH_NGEN=30 \
  nohup bash incremental_loop.sh > ../incr.log 2>&1 &
echo "Training on GPU 1, PID=$!"

# Start MD on GPU 0 (override GPUFLAGS to pin to device 0)
sleep 2
for id in 7K9D 2LBS 6TPH 2HGH; do
  CUDA_VISIBLE_DEVICES=0 nohup bash 02_run_production.sh $id > ../md/$id/prod_run.log 2>&1 &
  echo "MD $id on GPU 0, PID=$!"
done

sleep 3
echo "=== GPU status ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader
