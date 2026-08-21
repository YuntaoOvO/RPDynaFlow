#!/usr/bin/env bash
# ============================================================
# DynaFlow pipeline — shared configuration
# All scripts source this file. Override anything via env vars
# before calling, e.g.:  PROD_NS=30 bash 03_gpu_queue.sh
# ============================================================

# --- GROMACS discovery ----------------------------------------
# If gmx is not on PATH (non-interactive shells), source a GMXRC.
# Set GMXRC (e.g. /opt/gromacs/bin/GMXRC) or GMXBIN if your install is elsewhere.
if ! command -v gmx >/dev/null 2>&1; then
    if [ -n "${GMXRC:-}" ] && [ -f "$GMXRC" ]; then
        set +u; source "$GMXRC" >/dev/null 2>&1; set -u
    else
        for g in /opt/gromacs /usr/local/gromacs; do
            if [ -f "$g/bin/GMXRC" ]; then
                set +u; source "$g/bin/GMXRC" >/dev/null 2>&1; set -u
                break
            fi
        done
    fi
fi
# optional extra library dirs (PLUMED/FFTW/OpenMPI from custom builds)
if [ -n "${EXTRA_LIB_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$EXTRA_LIB_PATH:${LD_LIBRARY_PATH:-}"
fi

# --- paths ---------------------------------------------------
# ROOT = the DynaFlow folder that contains both "pipeline/" and
# the "RNA-protein complexes/" data directory.
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-$(dirname "$PIPELINE_DIR")}"
export PDB_DIR="${PDB_DIR:-$ROOT/RNA-protein complexes}"
export MD_DIR="${MD_DIR:-$ROOT/md}"
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
export MDP_DIR="$PIPELINE_DIR/mdp"

mkdir -p "$MD_DIR" "$DATA_DIR" "$RESULTS_DIR"

# --- force field ----------------------------------------------
# charmm36-jul2022 is the primary choice (good protein+RNA, one FF).
# It is NOT bundled with GROMACS: download once (see README §0).
# Fallback: amber99sb-ildn (bundled, weaker RNA) — set FF=amber99sb-ildn WATER=tip3p
export FF="${FF:-charmm36-jul2022}"
export WATER="${WATER:-tip3p}"
export SALT_M="${SALT_M:-0.15}"     # NaCl concentration
export BOX_D="${BOX_D:-1.0}"        # solute-to-box distance (nm)

# --- simulation length ----------------------------------------
export PROD_NS="${PROD_NS:-40}"     # production length per complex
export FRAME_PS="${FRAME_PS:-100}"  # frame saving interval

# --- GPU / threading ------------------------------------------
# 3 concurrent mdrun on the single GPU, 5 OpenMP threads each
# (16 vCPU total, leaving 1 for the OS).
export N_SLOTS="${N_SLOTS:-3}"
export NTOMP="${NTOMP:-5}"

# Detect CUDA build of GROMACS once; scripts use $GPUFLAGS. Respect a pre-set
# GPUFLAGS (e.g. GPUFLAGS="-nb cpu" to run MD on CPU while a training job hogs
# the GPU — uses idle vCPU, no conflict with the GPU job).
if [ -z "${GPUFLAGS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        export HAVE_GPU=1
        export GPUFLAGS="-nb gpu -pme gpu -bonded gpu -update gpu"
        export GPUFLAGS_PREP="-nb gpu -pme gpu -bonded gpu"
    else
        export HAVE_GPU=0
        export GPUFLAGS="-nb cpu"
        echo "[env.sh] WARNING: no GPU detected -> CPU-only mode. Reduce scope!" >&2
    fi
fi

# no GROMACS #backup.1# files
export GMX_MAXBACKUP=-1
# NOTE: charmm36-jul2022.ff lives in the GROMACS install share/top dir.
# Do NOT also set GMXLIB to a dir containing it — pdb2gmx aborts on
# duplicate FF matches. See download_ff.sh for fetching the force field.

# sanity: gmx must exist
if ! command -v gmx >/dev/null 2>&1; then
    echo "[env.sh] ERROR: 'gmx' not found. Load GROMACS first (source /path/to/gromacs/bin/GMXRC)" >&2
    return 1 2>/dev/null || exit 1
fi
