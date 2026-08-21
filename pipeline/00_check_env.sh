#!/usr/bin/env bash
# 00_check_env.sh — run FIRST. Verifies GROMACS, GPU, FF, python deps.
set -u
source "$(dirname "$0")/env.sh" || exit 1

echo "================ GROMACS ================"
gmx --version | head -5
if gmx --version 2>&1 | grep -qi "GPU support.*CUDA"; then
    echo "[OK] CUDA-enabled GROMACS build"
else
    echo "[!!] GROMACS looks like a CPU-only build. MD will be far too slow."
fi

echo "================ GPU ===================="
if [ "${HAVE_GPU:-0}" = 1 ]; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
    echo "[!!] No NVIDIA GPU visible"
fi

echo "============= force field ==============="
FF_FOUND=0
for d in . "$ROOT" "$GMXDATA" /usr/local/gromacs/share/gromacs/top /opt/gromacs*/share/gromacs/top; do
    [ -n "$d" ] && [ -d "$d/$FF.ff" ] && { echo "[OK] found $FF.ff in $d"; FF_FOUND=1; break; }
done
if [ "$FF_FOUND" = 0 ]; then
    echo "[!!] $FF.ff not found. Get it (one command, ~40 MB):"
    echo "     cd $ROOT && wget http://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2022.ff.tgz -O charmm36-jul2022.ff.tgz && tar xzf charmm36-jul2022.ff.tgz"
    echo "     (or fall back: export FF=amber99sb-ildn WATER=tip3p — bundled, weaker for RNA)"
fi

echo "============== python deps =============="
python3 - <<'EOF'
import importlib, sys
need = ["numpy", "pandas", "matplotlib", "MDAnalysis", "torch"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"[OK] {m:12s} {getattr(mod,'__version__','?')}")
    except ImportError:
        missing.append(m)
        print(f"[!!] {m} MISSING")
if "torch" not in missing:
    import torch
    print(f"     torch CUDA available: {torch.cuda.is_available()}")
if missing:
    print("install with: pip install " + " ".join(missing))
    sys.exit(1)
EOF

echo "============= disk / cores =============="
df -h "$ROOT" | tail -1
nproc
echo "========================================="
echo "If all [OK], proceed to: python3 00_select_systems.py"
