#!/usr/bin/env bash
# 01_prep_system.sh <PDBID> — clean PDB, build topology, solvate, ionize,
# minimize, NVT, NPT. After this, system is ready for 02_run_production.sh.
# Safe to re-run: skips steps whose output already exists.
set -euo pipefail
source "$(dirname "$0")/env.sh"

ID="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
LOWER="$(echo "$ID" | tr '[:upper:]' '[:lower:]')"
SRC_PDB="$(ls "$PDB_DIR/$ID"/*.pdb 2>/dev/null | grep -v '/\._' | head -1 || true)"
[ -z "$SRC_PDB" ] && SRC_PDB="$(ls "$PDB_DIR/$LOWER"/*.pdb 2>/dev/null | grep -v '/\._' | head -1 || true)"
[ -z "$SRC_PDB" ] && { echo "[$ID] ERROR: no PDB found under $PDB_DIR"; exit 1; }

W="$MD_DIR/$ID"
mkdir -p "$W"
cd "$W"
echo "=== [$ID] prep in $W (FF=$FF) ==="

# ---- 1. cleaning: model 1 only, standard ATOM records only ----------------
# Also remediates PDBv3 -> CHARMM36 atom names for nucleic-acid phosphate
# oxygens (OP1/OP2 -> O1P/O2P); CHARMM36 rtp uses the old names and pdb2gmx
# otherwise errors "Atom OP1 ... not found in rtp entry GUA".
# OP3 (5'-terminal extra phosphate oxygen) is also removed — 5TER patch handles the terminus.
if [ ! -s model1.pdb ]; then
    awk '
        /^MODEL/        {m++}
        /^ENDMDL/       {if (m==1) exit}
        /^ATOM/ {
            if (m<=1) {                                   # m==0 -> no MODEL records
                an = substr($0, 13, 4)
                if (an ~ /OP3/) next
                if (an ~ /OP1/ || an ~ /OP2/) {
                    gsub(/OP1/, "O1P", an); gsub(/OP2/, "O2P", an)
                    print substr($0, 1, 12) an substr($0, 17); next
                }
                print
            }
        }
        /^TER/          {if (m<=1) print}
    ' "$SRC_PDB" > model1.pdb
    [ -s model1.pdb ] || { echo "[$ID] ERROR: cleaning produced empty file"; exit 1; }
fi

# ---- 2. topology -----------------------------------------------------------
# CHARMM36 represents nucleic-acid termini as terminal PATCHES (5TER/3TER),
# not separate rtp building blocks, so pdb2gmx does NOT auto-select them — the
# default applies the protein patch NH3+ to the 5' nucleotide and crashes
# ("atom N not found in building block 1GUA"). Drive pdb2gmx with -ter via a
# python/pexpect wrapper that selects termini by residue (protein -> NH3+/COO-,
# nucleic -> 5TER/3TER). See pdb2gmx_auto.py. (env.sh puts miniconda python3,
# which has pexpect, on PATH.)
if [ ! -s conf.gro ]; then
    python3 "$PIPELINE_DIR/pdb2gmx_auto.py" \
        gmx pdb2gmx -f model1.pdb -o conf.gro -p topol.top -ff "$FF" -water "$WATER" \
        -ignh -ter > pdb2gmx.log 2>&1 \
        || { echo "[$ID] FAIL pdb2gmx (see $W/pdb2gmx.log)"; exit 1; }
fi

# ---- 3. box + solvent + ions ----------------------------------------------
[ -s boxed.gro ] || gmx editconf -f conf.gro -o boxed.gro -c -d "$BOX_D" -bt dodecahedron >> prep.log 2>&1
[ -s solv.gro ]  || gmx solvate -cp boxed.gro -cs spc216.gro -o solv.gro -p topol.top >> prep.log 2>&1
if [ ! -s solv_ions.gro ]; then
    gmx grompp -f "$MDP_DIR/em.mdp" -c solv.gro -p topol.top -o ions.tpr -maxwarn 1 >> prep.log 2>&1
    echo SOL | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top \
        -pname NA -nname CL -conc "$SALT_M" -neutral >> prep.log 2>&1
fi

# ---- 4. index groups: Protein_RNA + Water_and_ions -------------------------
if [ ! -s index.ndx ]; then
    if printf '"Protein" | "RNA"\nq\n' | gmx make_ndx -f solv_ions.gro -o index.ndx >> prep.log 2>&1 \
       && grep -q "Protein_RNA" index.ndx; then
        echo "[$ID] index: Protein_RNA group OK"
    else
        # fallback: build both groups with gmx select (residue-name based)
        echo "[$ID] make_ndx grouping failed, trying gmx select"
        if gmx select -s solv_ions.gro -on index.ndx \
            -select 'Protein_RNA = protein or rna' \
                    'Water_and_ions = not (protein or rna)' >> prep.log 2>&1; then
            echo "[$ID] index: groups OK via gmx select"
        else
            echo "[$ID] WARNING: index groups failed -> tcoupl falls back to 'system'"
            rm -f index.ndx
        fi
    fi
fi
# per-system mdp copies; degrade to single tc group if no index
mkdir -p mdp_local
for f in nvt npt prod; do cp "$MDP_DIR/$f.mdp" "mdp_local/$f.mdp"; done
cp "$MDP_DIR/em.mdp" mdp_local/em.mdp
if [ ! -s index.ndx ]; then
    sed -i -e 's/^tc-grps.*/tc-grps = system/' -e 's/^tau_t.*/tau_t = 1.0/' \
        -e 's/^ref_t.*/ref_t = 300/' mdp_local/*.mdp
fi
GMX_N=""
[ -s index.ndx ] && GMX_N="-n index.ndx"

# ---- 5. energy minimization -------------------------------------------------
if [ ! -s em.gro ]; then
    gmx grompp -f mdp_local/em.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 1 >> prep.log 2>&1
    gmx mdrun -deffnm em -ntmpi 1 -ntomp "$NTOMP" -nb gpu >> prep.log 2>&1 \
        || { echo "[$ID] FAIL EM"; exit 1; }
fi

# ---- 6. NVT ------------------------------------------------------------------
if [ ! -s nvt.gro ]; then
    gmx grompp -f mdp_local/nvt.mdp -c em.gro -r em.gro -p topol.top $GMX_N -o nvt.tpr -maxwarn 1 >> prep.log 2>&1
    gmx mdrun -deffnm nvt -ntmpi 1 -ntomp "$NTOMP" ${GPUFLAGS_PREP:-$GPUFLAGS} >> prep.log 2>&1 \
        || { echo "[$ID] FAIL NVT"; exit 1; }
fi

# ---- 7. NPT ------------------------------------------------------------------
if [ ! -s npt.gro ]; then
    gmx grompp -f mdp_local/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top $GMX_N -o npt.tpr -maxwarn 1 >> prep.log 2>&1
    gmx mdrun -deffnm npt -ntmpi 1 -ntomp "$NTOMP" ${GPUFLAGS_PREP:-$GPUFLAGS} >> prep.log 2>&1 \
        || { echo "[$ID] FAIL NPT"; exit 1; }
fi

echo "=== [$ID] prep DONE -> ready for production ==="
