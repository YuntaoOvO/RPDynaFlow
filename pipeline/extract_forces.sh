#!/usr/bin/env bash
# extract_forces.sh <PDBID> — recompute per-frame forces from the production
# trajectory via `gmx mdrun -rerun` with a force-output tpr (nstfout=1).
# Output: md/<ID>/rerunf.trr (per-atom forces, full solvated system).
# Needed for the force-matching model (Model B). Runs on GPU (brief, ~1 min).
# Idempotent: skips if rerunf.trr already exists.
#
# Two gotchas (hit & fixed 2026-08-01):
#   1. production .mdp has nstfout=0 -> must add nstfout=1 to a rerun.mdp
#   2. tc-grps=Protein_RNA needs -n index.ndx in grompp
set -uo pipefail
source "$(dirname "$0")/env.sh"
ID="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
W="$MD_DIR/$ID"; cd "$W"
[ -s rerunf.trr ] && { echo "[$ID] rerunf.trr exists, skipping"; exit 0; }
[ -s prod.xtc ] || { echo "[$ID] ERROR: no prod.xtc (production not done)"; exit 1; }
cp mdp_local/prod_run.mdp rerun.mdp
grep -q "^nstfout" rerun.mdp || echo "nstfout = 1" >> rerun.mdp
NARG=$([ -s index.ndx ] && echo "-n index.ndx" || echo "")
GMX_MAXBACKUP=-1 gmx grompp -f rerun.mdp -c npt.gro -p topol.top $NARG -o rerun.tpr -maxwarn 2 > rerun_grompp.log 2>&1 \
    || { echo "[$ID] grompp FAIL (see $W/rerun_grompp.log)"; exit 1; }
GMX_MAXBACKUP=-1 gmx mdrun -rerun prod.xtc -s rerun.tpr -deffnm rerunf \
    -nb gpu -pme gpu -bonded gpu -ntmpi 1 -ntomp "${NTOMP:-4}" > rerunf.log 2>&1 \
    || { echo "[$ID] mdrun -rerun FAIL (see $W/rerunf.log)"; exit 1; }
echo "[$ID] forces -> $W/rerunf.trr"
