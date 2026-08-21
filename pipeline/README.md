# DynaFlow MD dataset pipeline

Training-data generation for flow-matching on protein–RNA conformational ensembles.
**Design rationale:** the binding constraint is GPU-hours, not storage — one
RTX 4090-class GPU gives ~500–800 ns/day aggregate with 3 concurrent runs.

| decision | value |
|---|---|
| MD systems | 20 prepped → **16 target** (small, NMR-dynamic first) |
| length | **40 ns** per complex, 1 replica (16×40 = 640 ns ≈ ~28 h GPU) |
| frames | every **100 ps** → 400 frames/trajectory, ~6.4k total (~3 GB) |
| split | **3 train / 13 test (19%/81%)** — complex-level, generalization test |
| force field | CHARMM36-jul2022 + CHARMM TIP3P, 0.15 M NaCl, 300 K |
| model | conditional flow matching on **all solute heavy atoms** (protein + RNA + interface ligands), conditioned on the static structure |

## Directory layout

```
RPDynaFlow/                  # repo root
  RNA-protein complexes/     # input PDBs (create, or point PDB_DIR at yours)
  pipeline/                  # these scripts
  md/<PDBID>/                # per-system GROMACS workdirs (created)
  data/{md,nmr}/<PDBID>.npz  # featurized trajectories / NMR ensembles
  results/                   # model checkpoints + benchmark figs/tables
```

## Day 1 — setup, cleaning, prep, launch production (≈6 h work + overnight GPU)

```bash
cd pipeline
pip install -r requirements.txt          # numpy pandas matplotlib MDAnalysis torch
source /path/to/gromacs/bin/GMXRC        # GROMACS 2025.2, CUDA build

# CHARMM36 is NOT bundled with GROMACS — fetch once (~40 MB) with the repo script:
cd .. && bash download_ff.sh && cd pipeline
# fallback if download impossible: export FF=amber99sb-ildn (bundled, weaker RNA)

bash 00_check_env.sh                     # all [OK] before continuing
python3 00_select_systems.py             # cleaning + selection + split
                                         # -> ../data/systems_manifest.csv  (check it!)

# prep all selected systems (clean→pdb2gmx→solvate→ions→EM→NVT→NPT), ~15 min each
for id in $(python3 - <<'EOF'
import csv
print(" ".join(r["pdb_id"] for r in csv.DictReader(open("../data/systems_manifest.csv"))
      if r["selected"]=="1"))
EOF
); do bash 01_prep_system.sh "$id" || echo "PREP FAIL $id"; done

# launch the production queue (3 concurrent on the GPU, train systems first)
nohup bash 03_gpu_queue.sh > queue.log 2>&1 &
```

## Day 2 — productions finish; postprocess; train when train-split data ready

```bash
python3 05_postprocess.py                          # featurize whatever is done; safe to re-run
python3 05_postprocess.py --only <TRAIN_IDS>       # or just the 3 train systems first
python3 flow_model.py --epochs 300                 # ~1–3 h on the 4090; can start once
                                                   # the 3 train trajectories are featurized
```

## Day 3 — benchmark + figures

```bash
python3 05_postprocess.py        # make sure all test systems are featurized
python3 07_benchmark.py --n-gen 200
# -> ../results/benchmark_summary.csv, ../results/figs/*.png
```

## Knobs (env vars, set before running)

| var | default | when to change |
|---|---|---|
| `PROD_NS` | 40 | `PROD_NS=30` if throughput < ~400 ns/day |
| `N_SLOTS` | 3 | 2 if systems are big (>80k atoms), 4 if tiny |
| `NTOMP` | 5 | threads per run (16 vCPU / 3 slots ≈ 5) |
| `FRAME_PS` | 100 | 50 for denser sampling (doubles storage, still tiny) |
| `FF` / `WATER` | charmm36-jul2022 / tip3p | amber99sb-ildn fallback |
| `--n-candidates` | 20 | in 00_select_systems.py |

## If things go wrong

- **CPU-only GROMACS discovered** → aggregate ~30 ns/day: reduce to
  `python3 00_select_systems.py --n-candidates 8`, `PROD_NS=20` (8×20=160 ns ≈ 2 days). Tight but survivable.
- **pdb2gmx fails** (missing atoms / modified residues): the system is skipped; the
  candidate list has 4 spares. Check `md/<ID>/pdb2gmx.log`.
- **Queue stalls**: logs at `md/<ID>/prod.log`, `queue.log`. Rerun `03_gpu_queue.sh`
  anytime — finished systems (`prod.gro` present) are skipped.
- **Trajectory blows up** (RNA leaving box, LINCS errors): visible in
  `results/qc_md.csv` (rmsd_max_A huge). Drop that system from the manifest
  (`selected=0`) and re-run 05/07.

## Known simplifications

1. One 40 ns replica per complex — captures local/breathing dynamics, not slow
   large-scale rearrangements. (NMR ensembles serve as orthogonal ground truth.)
2. No Mg²⁺ (slow exchange kinetics); 0.15 M NaCl only.
3. Model 1 of each NMR ensemble used as the MD starting structure.
4. Box clearance 1.0 nm; long flexible RNA tails may interact with periodic images.
