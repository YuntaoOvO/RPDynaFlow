#!/usr/bin/env python3
"""md2pdb.py — convert MD (or NMR) trajectory npz to viewable multimodel PDBs.

Same machinery as samples2pdb.py, but reads the `coords` key (all frames,
already PBC-fixed and trjconv-aligned by 05_postprocess.py) from
data/md/<ID>.npz or data/nmr/<ID>.npz instead of generated samples.
The B-factor column carries the per-atom ensemble RMSF for putty/spectrum
coloring. Use together with samples2pdb.py output for MD-vs-generated
visual comparison in PyMOL/ChimeraX.

Run:  python3 md2pdb.py --systems 2LBS,6TPH,2HGH,7K9D,4M4O
      python3 md2pdb.py --data-dir ../data/nmr --out ../results/pdb_nmr
"""
import argparse, os
from samples2pdb import npz_to_pdb, MD_DIR

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="", help="comma-sep PDB IDs (else all in data dir)")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "md"),
                    help="dir with <ID>.npz trajectory files (data/md or data/nmr)")
    ap.add_argument("--max-models", type=int, default=0,
                    help="cap models per PDB (0 = all; evenly spaced subset)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pdb_md"),
                    help="output dir")
    args = ap.parse_args()

    pids = ([s.strip().upper() for s in args.systems.split(",") if s.strip()]
            if args.systems else
            sorted(f.replace(".npz", "").upper()
                   for f in os.listdir(args.data_dir) if f.endswith(".npz")))
    os.makedirs(args.out, exist_ok=True)

    for pid in pids:
        npz_p = os.path.join(args.data_dir, f"{pid}.npz")
        gro_p = os.path.join(MD_DIR, pid, "solute.gro")
        if not os.path.exists(npz_p):
            print(f"[skip] {pid}: no {npz_p}"); continue
        if not os.path.exists(gro_p):
            print(f"[skip] {pid}: no {gro_p}"); continue
        tag = os.path.basename(os.path.normpath(args.data_dir))
        out_p = os.path.join(args.out, f"{pid}_{tag}.pdb")
        n_m, n_a = npz_to_pdb(npz_p, gro_p, out_p, args.max_models, key="coords")
        print(f"[pdb] {pid}: {n_m} models x {n_a} atoms -> {out_p}")


if __name__ == "__main__":
    main()
