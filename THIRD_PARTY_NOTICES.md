# Third-party notices

RPDynaFlow is developed against, compares with, or interoperates with the
following third-party projects. Their code and data are **not bundled** in this
repository; each remains under its own license.

| Project | Role here | License | Upstream |
|---|---|---|---|
| **BioEmu** | External baseline for protein-ensemble generation; BioEmu-style coverage/free-energy metric semantics in `09_dynbench.py` | MIT (Microsoft) | https://github.com/microsoft/bioemu |
| **bioemu-benchmarks** | Official benchmark harness (multiconf, md_emulation, folding ΔG); used via `BIOEMU_BENCH_DIR` / `BIOEMU_BENCH_ROOT` env vars, cloned separately | MIT (Microsoft) | https://github.com/microsoft/bioemu-benchmarks |
| **ESMDynamic** | External baseline for dynamic contact-map prediction; compared via `esmdynamic_overlay.py` (requires separate install) | MIT (ShuklaGroup, UIUC) | https://github.com/ShuklaGroup/esmdynamic |
| **ESM** | Foundation of ESMDynamic (archived upstream) | MIT (Meta AI / facebookresearch) | https://github.com/facebookresearch/esm |
| **DRPScore** | Optional RNA-binding prediction score used in exploratory analysis (`drpscore_*.py`) | See upstream | https://github.com/Zhaolab-GitHub/DRPScore_v1.0 |
| **GROMACS** | MD engine for training-data generation (stages 01–05) | LGPL-2.1 | https://www.gromacs.org |
| **CHARMM36 force field (charmm36-jul2022)** | Protein/RNA force field for MD; fetched at setup by `download_ff.sh`, **not** bundled | CHARMM academic license | https://mackerell.umaryland.edu/charmm_ff.shtml |
| **PyTorch, NumPy, pandas, SciPy, matplotlib, MDAnalysis, tqdm** | Python dependencies (see `environment.yml`) | BSD/MIT respective | PyPI |

## Note on checkpoints

`checkpoints/flow_model_r{5,10,15}.pt` were trained by the authors on
GROMACS/CHARMM36 MD trajectories of Protein Data Bank structures (see
`examples/systems_manifest.csv` for PDB IDs). PDB entries used as training or
example inputs are subject to the RCSB PDB data policies
(https://www.wwpdb.org/terms-of-use).
