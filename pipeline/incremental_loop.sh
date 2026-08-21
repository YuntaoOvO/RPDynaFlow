#!/usr/bin/env bash
# incremental_loop.sh (v2) — K=1..N data-scaling ablation, per-round benchmark.
#
# Each round: ensure fresh featurization of completed systems, continue-train
# from the last checkpoint on the growing pool, save flow_model_r{K}.pt, then
# benchmark it against the featurized HELD-OUT systems (completed but not yet
# trained). Resumable: ROUND counter + LAST_CKPT reconstruct from existing r*.pt.
#
# Featurization is ALWAYS from the complete trajectory: a system's npz is rebuilt
# if missing OR older than its prod.gro (`-ot`), which catches stale partial
# featurizations left over from sanity-checks.
#
# Usage:  EPOCHS_PER_ROUND=50 BENCH_NGEN=30 bash incremental_loop.sh
set -uo pipefail   # NOT -e: one failure must not abort the loop
source "$(dirname "$0")/env.sh"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-50}"
BENCH_NGEN="${BENCH_NGEN:-30}"      # smaller n_gen for per-round; final pass uses more
POLL_S="${POLL_S:-60}"

CKPT_DIR="$RESULTS_DIR/checkpoints"
STATE="$RESULTS_DIR/incr_state.txt"      # trained pdb_ids, one per line
ORDER_FILE="$RESULTS_DIR/incr_order.txt"
mkdir -p "$CKPT_DIR"; touch "$STATE"

# --- viable system order (selected + prepped), train first by solute_atoms ---
python3 - "$DATA_DIR/systems_manifest.csv" "$ORDER_FILE" <<'EOF'
import csv, os, sys
md = os.environ.get("MD_DIR", "")
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["selected"] == "1"]
rows.sort(key=lambda r: (0 if r["split"] == "train" else 1, int(r["solute_atoms"])))
viable = [r for r in rows if os.path.exists(os.path.join(md, r["pdb_id"], "npt.gro"))]
with open(sys.argv[2], "w") as fh:
    for r in viable:
        fh.write(r["pdb_id"] + "\n")
print(f"[order] viable {len(viable)}/{len(rows)}: {' '.join(r['pdb_id'] for r in viable)}")
EOF
mapfile -t ORDER < <(grep -v '^$' "$ORDER_FILE")
echo "[incr] ${#ORDER[@]} viable systems | epochs/round=$EPOCHS_PER_ROUND bench_ngen=$BENCH_NGEN"

# featurize from COMPLETE trajectory; rebuild if npz missing or stale (older than prod.gro)
featurize_fresh() {
    local id="$1" npz gro
    npz="$DATA_DIR/md/$id.npz"; gro="$MD_DIR/$id/prod.gro"
    [ -s "$gro" ] || return 1
    if [ ! -s "$npz" ] || [ "$npz" -ot "$gro" ]; then
        rm -f "$npz" "$MD_DIR/$id/solute.gro" "$MD_DIR/$id/solute_pbc.xtc" "$MD_DIR/$id/fit.xtc"
        python3 "$PIPELINE_DIR/05_postprocess.py" --only "$id" || return 1
    fi
    return 0
}

# resume ROUND + LAST_CKPT from existing checkpoints
ROUND=$(ls "$CKPT_DIR"/flow_model_r*.pt 2>/dev/null | wc -l | tr -d ' ')
if [ "$ROUND" -gt 0 ]; then
    LAST_CKPT="$CKPT_DIR/flow_model_r${ROUND}.pt"
    echo "[incr] resuming: $ROUND checkpoints exist, last=$LAST_CKPT, state=[$(grep -v '^$' "$STATE" | paste -sd, -)]"
else
    LAST_CKPT=""
fi

while true; do
    NDONE=$(grep -cv '^$' "$STATE" 2>/dev/null || true)
    if [ "$NDONE" -ge "${#ORDER[@]}" ]; then echo "[incr] all ${#ORDER[@]} trained — DONE"; break; fi

    # A. eager-featurize every completed system (fresh check) — populates held-out npz
    for id in "${ORDER[@]}"; do featurize_fresh "$id" || true; done

    # B. next untrained system (first in order with prod.gro)
    NEXT=""
    for id in "${ORDER[@]}"; do
        grep -qx "$id" "$STATE" && continue
        [ -s "$MD_DIR/$id/prod.gro" ] && { NEXT="$id"; break; }
    done
    if [ -z "$NEXT" ]; then
        NGRO=$(ls "$MD_DIR"/*/prod.gro 2>/dev/null | wc -l | tr -d ' ')
        echo "[incr] $(date '+%H:%M') waiting — $NDONE/${#ORDER[@]} trained, $NGRO prod.gro"
        sleep "$POLL_S"; continue
    fi

    ROUND=$((ROUND+1))
    echo "[incr] === ROUND $ROUND: $NEXT ==="
    featurize_fresh "$NEXT" || { echo "[incr] featurize FAIL $NEXT"; sleep "$POLL_S"; continue; }
    POOL_CSV=$( { grep -v '^$' "$STATE"; echo "$NEXT"; } | paste -sd, -)
    CKPT="$CKPT_DIR/flow_model_r${ROUND}.pt"
    RESUME_ARG=""; [ -n "$LAST_CKPT" ] && [ -s "$LAST_CKPT" ] && RESUME_ARG="--resume $LAST_CKPT"

    # C+D. train + benchmark under the shared training lock (FIFO with Model B:
    # whoever holds results/training.lock runs to completion; the other waits)
    if (
        flock 9
        python3 "$PIPELINE_DIR/flow_model.py" --epochs "$EPOCHS_PER_ROUND" \
            --systems "$NEXT" $RESUME_ARG --save "$CKPT" || exit 1
        echo "$NEXT" >> "$STATE"
        HELD=""
        for id in "${ORDER[@]}"; do
            grep -qx "$id" "$STATE" && continue
            [ -s "$DATA_DIR/md/$id.npz" ] && HELD="$HELD,$id"
        done
        HELD="${HELD#,}"
        if [ -n "$HELD" ]; then
            python3 "$PIPELINE_DIR/07_benchmark.py" --ckpt "$CKPT" --systems "$HELD" \
                --n-gen "$BENCH_NGEN" --out "$RESULTS_DIR/bench_flow_model_r${ROUND}.csv" \
                && echo "[incr] r$ROUND benchmarked vs held-out [$HELD]" \
                || echo "[incr] bench FAIL r$ROUND (non-fatal)"
        else
            echo "[incr] r$ROUND: no featurized held-out yet — skip benchmark"
        fi
    ) 9>"$RESULTS_DIR/training.lock"; then
        LAST_CKPT="$CKPT"
        echo "[incr] round $ROUND OK: pool=[$POOL_CSV] -> $CKPT"
    else
        echo "[incr] train FAIL r$ROUND — not advanced"; sleep 10; continue
    fi
done

echo "[incr] DONE — checkpoints + per-round bench_*.csv in $RESULTS_DIR/"
