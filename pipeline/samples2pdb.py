#!/usr/bin/env python3
"""samples2pdb.py — batch-convert generated sample npz to viewable multimodel PDBs.

Reuses the inference_demo.py approach: md/<ID>/solute.gro is the topology template
(same heavy-atom selection as 05_postprocess.py), so atom/residue/chain names are
preserved; each generated conformer becomes one MODEL. The B-factor column carries
the per-atom ensemble RMSF for putty/spectrum coloring in PyMOL/ChimeraX.

Run:  python3 samples2pdb.py --samples ../results/samples/flow_model_r15
      python3 samples2pdb.py --samples ... --systems 2LBS,6TPH --max-models 100
"""
import argparse, os
import numpy as np

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_DIR = os.environ.get("MD_DIR", os.path.join(ROOT, "md"))

# identical to 05_postprocess.py atoms_from_trajectory selection
SEL = ("not name H* and not type H "
       "and not resname HOH SOL TIP3 SPC NA CL SOD CLA CAL MG")


def npz_to_pdb(npz_path, gro_path, out_path, max_models=0, key="gen_coords"):
    import MDAnalysis as mda
    gen = np.load(npz_path)[key].astype(np.float64)
    u = mda.Universe(gro_path)
    atoms = u.select_atoms(SEL)
    assert atoms.n_atoms == gen.shape[1], \
        f"atom mismatch: gro {atoms.n_atoms} vs npz {gen.shape[1]}"
    # per-atom ensemble RMSF -> B-factor column (visualization coloring)
    rmsf = np.sqrt(((gen - gen.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))
    if not hasattr(u.atoms, 'tempfactors'):
        u.add_TopologyAttr('tempfactors', [0.0] * u.atoms.n_atoms)
    atoms.tempfactors = rmsf
    if max_models and gen.shape[0] > max_models:
        idx = np.linspace(0, gen.shape[0] - 1, max_models).round().astype(int)
        gen = gen[idx]
    with mda.Writer(out_path, atoms.n_atoms) as w:
        for i in range(gen.shape[0]):
            atoms.positions = gen[i]
            w.write(atoms)
    return gen.shape[0], atoms.n_atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="dir with <ID>.npz sample files")
    ap.add_argument("--systems", default="", help="comma-sep PDB IDs (else all in dir)")
    ap.add_argument("--max-models", type=int, default=0,
                    help="cap models per PDB (0 = all; evenly spaced subset)")
    ap.add_argument("--out", default="", help="output dir (else <samples>/pdb)")
    args = ap.parse_args()

    pids = ([s.strip().upper() for s in args.systems.split(",") if s.strip()]
            if args.systems else
            sorted(f.replace(".npz", "").upper()
                   for f in os.listdir(args.samples) if f.endswith(".npz")))
    outdir = args.out if args.out else os.path.join(args.samples, "pdb")
    os.makedirs(outdir, exist_ok=True)

    for pid in pids:
        npz_p = os.path.join(args.samples, f"{pid}.npz")
        gro_p = os.path.join(MD_DIR, pid, "solute.gro")
        if not os.path.exists(npz_p):
            print(f"[skip] {pid}: no {npz_p}"); continue
        if not os.path.exists(gro_p):
            print(f"[skip] {pid}: no {gro_p}"); continue
        out_p = os.path.join(outdir, f"{pid}.pdb")
        n_m, n_a = npz_to_pdb(npz_p, gro_p, out_p, args.max_models)
        print(f"[pdb] {pid}: {n_m} models x {n_a} atoms -> {out_p}")


if __name__ == "__main__":
    main()
