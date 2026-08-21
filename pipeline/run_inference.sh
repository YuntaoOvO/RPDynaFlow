#!/bin/bash
# run_inference.sh — sample ensembles for held-out test systems with the
# shipped r15 checkpoint, then compute the legacy benchmark metrics.
# NOTE: --systems defaults assume you have featurized MD data for those IDs
# under $DATA_DIR (i.e., you ran the MD pipeline). For zero-shot inference on
# arbitrary PDBs use featurize_static_pdb.py + gen_ensembles.py --static.
cd "$(dirname "$0")"
source env.sh
REPO_ROOT="$(cd .. && pwd)"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/flow_model_r15.pt}"
echo "=== Inference with $(basename "$CKPT"), n_gen=200 ==="
python3 07_benchmark.py \
    --ckpt "$CKPT" \
    --n-gen 200 \
    --dump-samples \
    --out "$RESULTS_DIR/bench_r15_dynbench.csv"
echo "=== Inference done ==="
ls -la "$RESULTS_DIR/samples/$(basename "${CKPT%.pt}")"/ 2>/dev/null || true
