#!/usr/bin/env python3
"""featurize_static_pdb.py — build data/static/<ID>.npz from a single PDB.

Parses heavy atoms (protein + RNA + interface ligands) with the same element /
residue typing as 05_postprocess.py. No GROMACS or MD required. Used for
zero-shot inference on structures outside the training manifest.

Run:
  python3 featurize_static_pdb.py --pdb ../RNA-protein\\ complexes/4W5N/4W5N.pdb --id 4W5N
  python3 featurize_static_pdb.py --id 4W5N   # auto-find PDB under RNA-protein complexes/
  python3 featurize_static_pdb.py --validate 1EKZ  # compare atom count vs data/md/1EKZ.npz
"""
import argparse
import os
import sys

import numpy as np

# Reuse typing maps from 05_postprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

_pp = import_module("05_postprocess")
_ELEM = _pp._ELEM
RESTYPE = _pp.RESTYPE
PDB_DIR = _pp.PDB_DIR
DATA = _pp.DATA
ROOT = _pp.ROOT

_SKIP_RES = {"HOH", "SOL", "TIP3", "SPC", "WAT", "NA", "CL", "SOD", "CLA", "CAL", "MG"}
_RNA_RENAME = {"OP1": "O1P", "OP2": "O2P"}  # PDB -> CHARMM-style (for name consistency)


def _find_pdb(pid):
    d = os.path.join(PDB_DIR, pid)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".pdb") and not f.startswith("."):
                return os.path.join(d, f)
    p = os.path.join(PDB_DIR, f"{pid.lower()}.pdb")
    if os.path.exists(p):
        return p
    p2 = os.path.join(PDB_DIR, f"{pid.upper()}.pdb")
    return p2 if os.path.exists(p2) else None


def parse_static_pdb(pdb_path, altloc="first"):
    """Parse MODEL 1 heavy atoms from a PDB file.

    Returns static (A,3), elem, residx, res_restype, atom_names, records
    where records is a list of dicts for PDB writing.
    """
    atoms = []
    in_model = False
    model_count = 0
    seen_alt = {}

    def _parse_coord(line):
        if len(line) > 54 and line[46:54].strip():
            return [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        return [float(line[30:38]), float(line[38:46]), float(line[46:54])]

    with open(pdb_path, errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                model_count += 1
                in_model = model_count == 1
                continue
            if rec == "ENDMDL":
                if model_count == 1:
                    break
                in_model = False
                continue
            if rec not in ("ATOM", "HETATM"):
                continue
            if not in_model and model_count == 0:
                pass  # no MODEL records — accept all ATOM/HETATM
            elif not in_model:
                continue

            name = line[12:16].strip()
            alt = line[16] if len(line) > 16 else " "
            resn = line[17:20].strip().upper()
            chain = line[21] if len(line) > 21 else " "
            resseq = line[22:26].strip()
            elem_ch = (line[76:78].strip() if len(line) > 76 else "") or name[:1]
            elem_ch = elem_ch.upper()

            if name[:1] == "H" or elem_ch == "H":
                continue
            if resn in _SKIP_RES:
                continue
            if alt not in (" ", "A") and altloc == "first":
                continue

            key = (chain, resseq, resn, name)
            if altloc == "first" and key in seen_alt:
                continue
            seen_alt[key] = True

            aname = _RNA_RENAME.get(name, name)
            xyz = _parse_coord(line)
            atoms.append(dict(
                chain=chain, resseq=resseq, resn=resn, name=aname,
                elem=elem_ch, xyz=xyz, line=line.rstrip("\n"),
            ))

    if not atoms:
        raise ValueError(f"no heavy atoms parsed from {pdb_path}")

    # Compact residue indexing in file order
    seen_res, res_order = {}, []
    for a in atoms:
        rk = (a["chain"], a["resseq"], a["resn"])
        if rk not in seen_res:
            seen_res[rk] = len(res_order)
            res_order.append(a["resn"])

    static = np.array([a["xyz"] for a in atoms], dtype=np.float32)
    elem = np.array([_ELEM.get(a["elem"], 5) for a in atoms], dtype=np.int64)
    residx = np.array([seen_res[(a["chain"], a["resseq"], a["resn"])] for a in atoms],
                      dtype=np.int64)
    res_rt = np.array([RESTYPE.get(rn, 23) for rn in res_order], dtype=np.int64)  # unknown->23
    atom_names = np.array([a["name"] for a in atoms])
    return static, elem, residx, res_rt, atom_names, atoms


def write_static_npz(pid, pdb_path, out_dir=None):
    out_dir = out_dir or os.path.join(DATA, "static")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{pid}.npz")

    static, elem, residx, res_rt, atom_names, _ = parse_static_pdb(pdb_path)
    coords = static[None].copy()
    np.savez_compressed(
        out,
        coords=coords,
        static=static,
        atom_elements=elem,
        atom_residx=residx,
        res_restype=res_rt,
        atom_names=atom_names,
        pdb_id=pid,
        source_pdb=os.path.abspath(pdb_path),
    )
    return out, int(static.shape[0])


def validate_against_md(pid):
    md_p = os.path.join(DATA, "md", f"{pid}.npz")
    if not os.path.exists(md_p):
        print(f"[validate] no MD npz for {pid}")
        return False
    pdb_path = _find_pdb(pid)
    if not pdb_path:
        print(f"[validate] no PDB for {pid}")
        return False
    static, elem, residx, res_rt, atom_names, _ = parse_static_pdb(pdb_path)
    z = np.load(md_p)
    ok = True
    for key, arr in [
        ("n_atoms", (len(static), z["static"].shape[0])),
        ("elem_hist", (np.bincount(elem, minlength=6), np.bincount(z["atom_elements"], minlength=6))),
        ("n_res", (len(res_rt), len(z["res_restype"]))),
    ]:
        if key == "n_atoms":
            match = arr[0] == arr[1]
            print(f"  atoms: pdb={arr[0]} md={arr[1]} {'OK' if match else 'MISMATCH'}")
            ok &= match
        elif key == "elem_hist":
            match = np.array_equal(arr[0], arr[1])
            print(f"  elem hist: {'OK' if match else 'MISMATCH'} pdb={arr[0]} md={arr[1]}")
            ok &= match
        else:
            match = arr[0] == arr[1]
            print(f"  residues: pdb={arr[0]} md={arr[1]} {'OK' if match else 'MISMATCH'}")
            ok &= match
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="", help="input PDB path")
    ap.add_argument("--id", default="", help="PDB ID (uppercase)")
    ap.add_argument("--validate", default="", help="compare parser to data/md/<ID>.npz")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    if args.validate:
        pid = args.validate.upper()
        print(f"[validate] {pid} vs MD npz")
        ok = validate_against_md(pid)
        sys.exit(0 if ok else 1)

    pid = args.id.upper() if args.id else ""
    pdb_path = args.pdb
    if not pdb_path and pid:
        pdb_path = _find_pdb(pid)
    if not pdb_path or not os.path.exists(pdb_path):
        ap.error("need --pdb or --id with PDB under RNA-protein complexes/")
    if not pid:
        pid = os.path.splitext(os.path.basename(pdb_path))[0].upper()

    out, n_atoms = write_static_npz(pid, pdb_path, args.out_dir or None)
    print(f"[static] {pid}: {n_atoms} heavy atoms -> {out}")


if __name__ == "__main__":
    main()
