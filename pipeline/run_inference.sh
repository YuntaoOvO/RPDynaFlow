#!/bin/bash
cd "$(dirname "$0")"
source env.sh
echo "=== Starting inference with r3, n_gen=200 ==="
python3 07_benchmark.py \
    --ckpt ../results/checkpoints/flow_model_r3.pt \
    --n-gen 200 \
    --dump-samples \
    --systems 1A1T,4PDB,1DK1,2Y8W,2XDB,6GBM,2FY1,2N82 \
    --out ../results/bench_r3_dynbench.csv
echo "=== Inference done ==="
ls -la ../results/samples/flow_model_r3/
