#!/usr/bin/env bash
# ff_patch_met_terminus.sh — fix the CHARMM36-jul2022 name collision that makes
# pdb2gmx silently abort (exit 1, "atom C1 not found in building block 1MET")
# on any protein chain whose N-terminus is methionine.
#
# Root cause (confirmed from pdb2gmx logs):
#   charmm36-jul2022.ff/ethers.n.tdb defines [ MET1 ] and ethers.c.tdb defines
#   [ MET2 ].  These are terminal patches for an ETHER residue (atoms C1/H1A/
#   H1B/O1, from top_all35_ethers.rtf), NOT methionine.  pdb2gmx matches the
#   "MET1" name to an N-terminal amino-acid MET and applies the wrong patch,
#   which then references atoms that don't exist in aminoacids.rtp [ MET ].
#   These patches are referenced by NO .r2b file, so renaming them is safe.
#
# Fix: rename the two patch headers so the name no longer collides; pdb2gmx then
# falls back to the standard NH3+ / COO- termini for amino-acid methionine.
#
#   [ MET1 ] -> [ EMET1 ]   (ethers.n.tdb)
#   [ MET2 ] -> [ EMET2 ]   (ethers.c.tdb)
#
# Idempotent (skips if already patched). Originals backed up to
# $ROOT/ff_patch_backup/ (a normal copy, NOT a GROMACS #backup file). Re-run
# after any GROMACS reinstall or instance rebuild — the GROMACS install dir is
# not on the persistent data disk and may be reset when the GPU is re-mounted.
set -euo pipefail
source "$(dirname "$0")/env.sh"   # puts gmx on PATH; sets ROOT

# locate charmm36-jul2022.ff (the dir gmx actually reads; GMXLIB is NOT set)
_GMX_BIN="$(command -v gmx || true)"
FFDIR=""
for d in "${FF_DIR:-}" \
         "${GMXDATA:-}" \
         "$([ -n "$_GMX_BIN" ] && dirname "$(dirname "$_GMX_BIN")")/share/gromacs/top" \
         /opt/gromacs/share/gromacs/top \
         /usr/local/gromacs/share/gromacs/top; do
    [ -n "$d" ] && [ -f "$d/charmm36-jul2022.ff/ethers.n.tdb" ] && { FFDIR="$d/charmm36-jul2022.ff"; break; }
done
[ -z "$FFDIR" ] && { echo "[ff_patch] ERROR: charmm36-jul2022.ff not found"; exit 1; }
echo "[ff_patch] force field: $FFDIR"

# idempotency check
NEED=0
grep -q '^\[ MET1 \]' "$FFDIR/ethers.n.tdb" 2>/dev/null && NEED=1
grep -q '^\[ MET2 \]' "$FFDIR/ethers.c.tdb" 2>/dev/null && NEED=1
if [ "$NEED" = 0 ]; then
    echo "[ff_patch] already patched (no bare [ MET1 ]/[ MET2 ]) — nothing to do"
    exit 0
fi

BK="$ROOT/ff_patch_backup"
mkdir -p "$BK"
cp -p "$FFDIR/ethers.n.tdb" "$BK/ethers.n.tdb.orig"
cp -p "$FFDIR/ethers.c.tdb" "$BK/ethers.c.tdb.orig"
sed -i 's/^\[ MET1 \]/[ EMET1 ]/' "$FFDIR/ethers.n.tdb"
sed -i 's/^\[ MET2 \]/[ EMET2 ]/' "$FFDIR/ethers.c.tdb"

# verify
grep -q '^\[ MET1 \]' "$FFDIR/ethers.n.tdb" && { echo "[ff_patch] ERROR: [ MET1 ] still present"; exit 1; }
grep -q '^\[ MET2 \]' "$FFDIR/ethers.c.tdb" && { echo "[ff_patch] ERROR: [ MET2 ] still present"; exit 1; }
echo "[ff_patch] renamed [ MET1 ]->[ EMET1 ], [ MET2 ]->[ EMET2 ]; originals in $BK/"
echo "[ff_patch] OK"
