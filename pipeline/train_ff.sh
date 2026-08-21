#!/usr/bin/env bash
# train_ff.sh — Model B trainer (force/energy). Acquires the shared training.lock
# (flock) so it NEVER overlaps Model A's rounds (FIFO, no preemption); runs as ONE
# continuous flow_model_ff.py (no chunking — clear epoch progress). Starts only when
# the slot is free (A idle); A's incoming rounds queue behind this run.
#
# Usage:  EPOCHS=150 bash train_ff.sh   (150 ep × 15 systems = same total steps as A r15)
#         nohup bash train_ff.sh > results/ff/train.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
EPOCHS="${EPOCHS:-150}"
SYSTEMS="${SYSTEMS:-1NYB,2ESE,1EKZ,1A1T,4PDB,1DK1,2XDB,2Y8W,6GBM,2FY1,2N82,2L2K,2N3O,1RKJ,1FJE}"
LAM_F="${LAM_F:-1.0}"     # force-matching weight (primary physics signal)
LAM_E="${LAM_E:-0.1}"     # energy weight (small; full-system caveat)
LOCK="$RESULTS_DIR/training.lock"
mkdir -p "$RESULTS_DIR/ff"
echo "[ff] waiting for training slot (lock: $LOCK) ..."
(
    flock 9
    echo "[ff] slot acquired @ $(date '+%H:%M') — training Model B" \
         "(${EPOCHS} ep, systems=${SYSTEMS}, λ_f=${LAM_F}, λ_e=${LAM_E})"
    python3 "$PIPELINE_DIR/flow_model_ff.py" \
        --epochs "$EPOCHS" --systems "$SYSTEMS" \
        --lambda-f "$LAM_F" --lambda-e "$LAM_E"
    echo "[ff] Model B done @ $(date '+%H:%M')"
) 9>"$LOCK"
echo "[ff] released slot"
