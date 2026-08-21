# RPDynaFlow

**RPDynaFlow** generates conformational ensembles of protein–RNA complexes from a single static structure via atomic conditional flow matching (CFM).

Given one experimental structure (crystal, cryo-EM, or NMR model 1), the model samples alternative conformations of **all solute heavy atoms** (protein + RNA + mediating ligands/ions) in the same aligned frame — no MD, no MSA, no topology required at inference.

```mermaid
flowchart LR
  pdb[Static PDB] --> feat[Heavy-atom npz]
  md[Short MD] --> feat
  feat --> graph[Fixed 8A neighbor graph]
  graph --> cfm[AtomFlowNet CFM]
  cfm --> ens[Generated ensemble]
```

**Approach:** train on short (40 ns) MD trajectories started from the experimental structure — the core split uses just **3 small NMR training complexes**, and the shipped checkpoints scale that to 5 / 10 / 15 systems (`r5` / `r10` / `r15`) — then generalize to held-out systems, including zero-shot transfer to much larger complexes (~4.2k–6.8k heavy atoms vs ~1.1k–3.4k in training).

## Repository layout

```
RPDynaFlow/
├── pipeline/          # all scripts (numbered stages + models + benchmarks)
├── checkpoints/       # pretrained CFM checkpoints (r5 / r10 / r15)
├── examples/          # Ago2 zero-shot example inputs (4W5N, 9K6T)
├── environment.yml    # conda environment spec
└── download_ff.sh     # CHARMM36 fetch (only for MD training-data generation)
```

## Quickstart (inference)

```bash
# 1) environment
conda env create -f environment.yml
conda activate rpdynaflow

# 2) run the smoke test: sample 10 conformers of Ago2 (4W5N) on GPU
cd pipeline
bash smoke_test.sh          # ~1 min on an 8 GB GPU

# 3) sample a full ensemble (200 conformers)
export DATA_DIR=$PWD/../examples
python3 gen_ensembles.py \
    --ckpt ../checkpoints/flow_model_r15.pt \
    --systems 4W5N \
    --n-gen 200 --static \
    --out /tmp/r15_4W5N
```

The output npz contains `gen (S,A,3)` sampled coordinates plus the static frame and atom metadata, and a viewable multi-model PDB is written alongside; pass `--no-pdb` to skip it. `featurize_static_pdb.py --pdb <path> --id <PDBID>` converts any new PDB into the same npz format (see below).

## Model (AtomFlowNet + CFM)

Implementation: [`pipeline/flow_model.py`](pipeline/flow_model.py)

| Component | Detail |
|---|---|
| Nodes | All solute heavy atoms (protein + RNA + mediating ligands/ions) |
| Equivariance | None (aligned-frame displacement regression) |
| Edges | Pairs within **8 Å** on the **static** structure (fixed graph per system) |
| Edge features | Gaussian RBF of static inter-atomic distance (16 bins, cutoff-smoothed) |
| Backbone | SchNet-style interaction blocks (`d=128`, 5 layers, ~0.3M params) |
| State | `x1 = (coords − static) / SCALE`, `SCALE = 10` Å |
| CFM | `x_t = (1−t)·z + t·x1`, target velocity `v* = x1 − z`, `z ~ N(0, σ²)` |
| Sampling | Euler integration of the learned velocity field (30 steps default) |

Training draws one system per step and several MD frames; systems are never padded together. `σ` is the per-dataset displacement std computed once at train start.

### Model variants

| ID | Description | Script / checkpoint |
|---|---|---|
| **A** | CFM only | `flow_model.py` → `checkpoints/flow_model_r{5,10,15}.pt` |
| **B** | A + force/energy matching | `flow_model_ff.py` → `results/ff/flow_model_ff.pt` (train yourself) |
| **C** | B + energy-guided sampling (`η`) | same B checkpoint, `07_benchmark.py --guided` |

## Data format (`*.npz`)

Featurizer: [`pipeline/05_postprocess.py`](pipeline/05_postprocess.py) (from MD) or [`pipeline/featurize_static_pdb.py`](pipeline/featurize_static_pdb.py) (PDB-only, no MD).

| Key | Shape | Meaning |
|---|---|---|
| `coords` | `(F, A, 3)` | Per-frame heavy-atom coordinates (Å) |
| `static` | `(A, 3)` | Conditioning structure (frame 0); `F=1` when `coords == static` |
| `atom_elements` | `(A,)` | Class 0–5: C, N, O, P, S, other (**class indices, not atomic numbers**) |
| `atom_residx` | `(A,)` | Residue index per atom |
| `res_restype` | `(Nres,)` | 0–19 amino acids, 20–23 A/U/G/C, 24 other |
| `atom_names` | `(A,)` | Atom names (for benchmarks/PDB writing) |

`examples/static/<PDBID>.npz` has `F=1` (`coords == static`) — zero-shot inference mode; no MD or GROMACS topology required.

## Pipeline stages

| Script | Purpose | Output |
|---|---|---|
| `00_select_systems.py` | System selection + train/test split | `data/systems_manifest.csv` |
| `01`–`03` | GROMACS prep + 40 ns production | `md/<ID>/prod.xtc` |
| `05_postprocess.py` | trjconv + featurize | `data/md/<ID>.npz` |
| `flow_model.py` | Train Model A | `results/checkpoints/flow_model_r{K}.pt` |
| `incremental_loop.sh` | K=1→15 data-scaling ablation | |
| `gen_ensembles.py` | Sample ensembles for benchmarks / PyMOL | `results/samples/...` |
| `07_benchmark.py` | Legacy metrics (RMSF Pearson, coverage, JS, …) | |
| `09_dynbench.py` | Tiered dynamic benchmark (DCCM, RMSIP, BioEmu-style coverage, FE-MAE) | |

**GPU required for inference** — the GNN forward pass hangs on CPU.

## Retraining from scratch (MD data generation)

The full pipeline that produced the training data uses GROMACS + CHARMM36:

```bash
# 0) force field (~40 MB, not bundled): fetch into the repo root
bash download_ff.sh                  # → charmm36-jul2022.ff/

cd pipeline
source env.sh                        # needs gmx on PATH (or set GMXRC=/path/to/GMXRC)
bash 00_check_env.sh                 # all [OK] before continuing
python3 00_select_systems.py         # cleaning + selection + split
bash 01_prep_system.sh <PDBID>       # pdb2gmx + solvate + ions + EM
bash 03_gpu_queue.sh                 # 40 ns production × N systems
python3 05_postprocess.py --only <ID>   # gmx trjconv + featurize → data/md/<ID>.npz
python3 flow_model.py                # train Model A
```

CHARMM36 RNA termini need `ff_patch_met_terminus.sh` + `pdb2gmx_auto.py` (included). The 15-system manifest is `examples/systems_manifest.csv`.

## Benchmarks

Internal dynamic benchmark ([`pipeline/09_dynbench.py`](pipeline/09_dynbench.py)):

| Tier | Metrics |
|---|---|
| A | RMSF Pearson/Spearman, mean-RMSF ratio vs MD block CI, RMSD Wasserstein |
| B | DCCM, RMSIP (top-10 PCs), contact frequency (incl. RNA–protein) |
| C | PCA macrostate JS, KDE Bhattacharyya, BioEmu-style MD-frame coverage, 2D free-energy MAE |
| D | Clash rate, bond-length sanity |

A **null Gaussian** baseline (per-atom noise with MD-matched RMSF) is included.

### External comparisons

| Method | Scope | Use here |
|---|---|---|
| [BioEmu](https://github.com/microsoft/bioemu) | Protein ensembles from sequence | Protein-only baseline; coverage/FE semantics via `bioemu_*.py` |
| [bioemu-benchmarks](https://github.com/microsoft/bioemu-benchmarks) | multiconf, md_emulation, folding ΔG | clone it, then `dynaflow_bench_adapter.py` / `bioemu_evaluate.py` (`BIOEMU_BENCH_DIR` env var) |
| [ESMDynamic](https://github.com/ShuklaGroup/esmdynamic) | Dynamic contact maps from sequence | contact-frequency overlay via `esmdynamic_overlay.py` (separate install) |

## Ago2 zero-shot showcase (4W5N vs 9K6T)

Human Argonaute-2 — guide-only (4W5N) vs guide+target (9K6T) — a zero-shot test outside the 15-system MD training set:

```bash
cd pipeline
export DATA_DIR=$PWD/../examples
export PDB_DIR=$PWD/../examples/pdb
python3 gen_ensembles.py --ckpt ../checkpoints/flow_model_r15.pt \
    --systems 4W5N,9K6T --n-gen 200 --static --out ../results/samples/ago2
python3 ago2_multiconf_eval.py --samples-dir ../results/samples/ago2 \
    # cross-structure RMSD coverage on aligned domains (defaults: 4W5N vs 9K6T)
```

**Known limits** (be aware when interpreting outputs): training data covers small motif fluctuations in the bound basin; large Ago2 domain motions may exceed the learned amplitude, and the fixed 8 Å graph cannot rebuild edges after large rigid-body moves. Cross-structure metrics use sequence-aligned protein Cα (plus shared guide nucleotides where applicable), not full complex atom sets.

## License

Code: MIT (see [LICENSE](LICENSE)). CHARMM36 force field has its own license — fetched separately by `download_ff.sh`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for benchmark/comparison-code provenance.

## Citation

If you use RPDynaFlow, please cite this repository (`https://github.com/YuntaoOvO/RPDynaFlow`) and watch for the accompanying preprint.
