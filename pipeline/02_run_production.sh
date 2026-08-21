#!/usr/bin/env bash
# 02_run_production.sh <PDBID> [pinoffset] — production MD for one system.
# Called by 03_gpu_queue.sh (or directly). Skips if prod.gro exists.
set -euo pipefail
source "$(dirname "$0")/env.sh"

ID="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
PINOFF="${2:-0}"
W="$MD_DIR/$ID"
cd "$W"

[ -s prod.gro ] && { echo "[$ID] production already done, skipping"; exit 0; }
[ -s npt.gro ] || { echo "[$ID] ERROR: npt.gro missing — run 01_prep_system.sh first"; exit 1; }

NSTEPS=$(( PROD_NS * 500000 ))          # dt=2fs -> 500000 steps/ns
NSTXOUT=$(( FRAME_PS * 500 ))           # dt=2fs -> 500 steps/ps
sed -e "s/__NSTEPS__/$NSTEPS/" -e "s/__NSTXOUT__/$NSTXOUT/" \
    mdp_local/prod.mdp > mdp_local/prod_run.mdp

GMX_N=""
[ -s index.ndx ] && GMX_N="-n index.ndx"

if [ ! -s prod.tpr ]; then
    gmx grompp -f mdp_local/prod_run.mdp -c npt.gro -t npt.cpt -p topol.top $GMX_N \
        -o prod.tpr -maxwarn 1 >> prep.log 2>&1
fi

echo "[$ID] production ${PROD_NS} ns starting (pinoffset=$PINOFF, GPU=$HAVE_GPU)"
# Resume from checkpoint if present: a killed run leaves prod.cpt; WITHOUT -cpi, mdrun
# restarts from step 0 (this once lost 1A1T's 73% progress). -cpi appends safely and
# is a no-op for a fresh system (no cpt -> empty $CPI).
CPI=$([ -s prod.cpt ] && echo "-cpi prod.cpt" || echo "")
gmx mdrun -deffnm prod $CPI -ntomp "$NTOMP" -pin on -pinoffset "$PINOFF" \
    $GPUFLAGS -maxh "${PROD_MAXH:-48}" >> prod.log 2>&1 \
    || { echo "[$ID] FAIL production (see $W/prod.log)"; exit 1; }

echo "[$ID] production DONE: $(ls -lh prod.xtc | awk '{print $5}') xtc"
