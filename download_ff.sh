#!/usr/bin/env bash
# download_ff.sh — fetch the CHARMM36 (July 2022) protein/RNA force field.
# Only needed for MD training-data generation (pipeline stages 01–05);
# inference (featurize_static_pdb + gen_ensembles) does NOT need it.
#
# Usage:  bash download_ff.sh          # extracts ./charmm36-jul2022.ff next to this script
#         FF_INSTALL_DIR=/some/gromacs/share/gromacs/top bash download_ff.sh
#
# License: CHARMM academic license — see https://mackerell.umaryland.edu/charmm_ff.shtml
set -euo pipefail

VERSION=charmm36-jul2022
URL="https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/${VERSION}.ff.tgz"
DEST="${FF_INSTALL_DIR:-$(dirname "$0")}"

if [ -f "${DEST}/${VERSION}.ff/ethers.n.tdb" ]; then
    echo "[download_ff] ${DEST}/${VERSION}.ff already present — nothing to do."
    exit 0
fi

echo "[download_ff] fetching ${VERSION}.ff.tgz (~40 MB)"
wget -q --show-progress "${URL}" -O "${DEST}/${VERSION}.ff.tgz" \
  || curl -fL "${URL}" -o "${DEST}/${VERSION}.ff.tgz"

tar -xzf "${DEST}/${VERSION}.ff.tgz" -C "${DEST}"
rm -f "${DEST}/${VERSION}.ff.tgz"

# If installed into a GROMACS top dir, patch RNA termini (idempotent):
if [ -n "${FF_INSTALL_DIR:-}" ] && command -v bash >/dev/null; then
    FF_DIR="${DEST}/${VERSION}.ff" "$(dirname "$0")/pipeline/ff_patch_met_terminus.sh" || true
fi
echo "[download_ff] done: ${DEST}/${VERSION}.ff"
