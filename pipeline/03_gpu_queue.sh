#!/usr/bin/env bash
# 03_gpu_queue.sh — run productions for all selected systems, N_SLOTS at a
# time on the single GPU. Train systems go FIRST so featurization/training
# can start while test systems finish.
# Usage:  bash 03_gpu_queue.sh            (foreground)
#         nohup bash 03_gpu_queue.sh > queue.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/env.sh"

MANIFEST="$DATA_DIR/systems_manifest.csv"
[ -s "$MANIFEST" ] || { echo "manifest missing — run 00_select_systems.py first"; exit 1; }

# build ordered work list: selected=1, train first, then by solute_atoms
mapfile -t WORK < <(python3 - "$MANIFEST" <<'EOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["selected"] == "1"]
rows.sort(key=lambda r: (0 if r["split"] == "train" else 1, int(r["solute_atoms"])))
for r in rows: print(r["pdb_id"])
EOF
)
echo "queue: ${#WORK[@]} systems -> ${WORK[*]}"

PIN_OFFS=(); for ((i=0; i<N_SLOTS; i++)); do PIN_OFFS+=($((i * NTOMP))); done
declare -A SLOT_PID=()

launch() {  # $1=ID  $2=slot
    local id="$1" slot="$2"
    nohup bash "$PIPELINE_DIR/02_run_production.sh" "$id" "${PIN_OFFS[$slot]}" \
        > "$MD_DIR/$id/queue_run.log" 2>&1 &
    SLOT_PID[$slot]=$!
    echo "[queue] slot $slot -> $id (pid ${SLOT_PID[$slot]})"
}

IDX=0
while [ $IDX -lt ${#WORK[@]} ] || [ ${#SLOT_PID[@]} -gt 0 ]; do
    # fill free slots
    for ((s=0; s<N_SLOTS; s++)); do
        [ $IDX -ge ${#WORK[@]} ] && break
        if [ -z "${SLOT_PID[$s]:-}" ]; then
            ID="${WORK[$IDX]}"
            if [ -s "$MD_DIR/$ID/prod.gro" ]; then
                echo "[queue] $ID already done"; IDX=$((IDX+1)); continue
            fi
            [ -s "$MD_DIR/$ID/npt.gro" ] || {
                echo "[queue] $ID not prepped, skipping (check 01_prep_system.sh)"; IDX=$((IDX+1)); continue; }
            launch "$ID" "$s"; IDX=$((IDX+1))
        fi
    done
    # reap finished slots
    sleep 60
    for s in "${!SLOT_PID[@]}"; do
        if ! kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
            unset "SLOT_PID[$s]"
        fi
    done
done
echo "[queue] ALL DONE"
