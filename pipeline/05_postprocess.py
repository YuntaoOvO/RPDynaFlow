#!/usr/bin/env python3
"""05_postprocess.py — atomic featurizer (all heavy atoms: protein + RNA + ligands).

All trajectory processing is done by gmx trjconv UPSTREAM:
    -pbc cluster -center -ur compact   (PBC / centering; cluster, NOT mol — mol split
                                        protein/RNA to opposite box faces in ~25% of frames)
    -fit rot+trans                     (align every frame to the start structure)
This script reads the clean aligned trajectory and writes per-atom coordinates:
    coords       (F,A,3)   per-frame heavy-atom xyz, gmx-aligned (A)
    static       (A,3)     frame-0 (the conditioning structure)
    atom_elements(A,)      element class (C,N,O,P,S,other = 0..5)
    atom_residx  (A,)      residue index of each atom (into res_restype)
    res_restype  (Nres,)   0-19 aa, 20-23 A/U/G/C, 24 other(ligand)
    pdb_id
Ligands/ions that mediate RNA interactions are KEPT (not dropped); only bulk
water and free Na/Cl are excluded. flow_model.py builds the geometry graph.
"""
import argparse, os, subprocess
import numpy as np

os.environ["GMX_MAXBACKUP"] = "-1"
for _g in (os.environ.get("GMXBIN", ""), "/opt/gromacs/bin", "/usr/local/gromacs/bin"):
    if _g and os.path.isdir(_g) and _g not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _g + os.pathsep + os.environ["PATH"]

AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
NT = ["A", "U", "G", "C"]
RESTYPE = {r: i for i, r in enumerate(AA)}
RESTYPE.update({r: 20 + i for i, r in enumerate(NT)})
_ELEM = {"H": 5, "C": 0, "N": 1, "O": 2, "P": 3, "S": 4}  # -> class (other=5)

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_DIR = os.environ.get("MD_DIR", os.path.join(ROOT, "md"))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
PDB_DIR = os.environ.get("PDB_DIR", os.path.join(ROOT, "RNA-protein complexes"))
# Fresh-clone fallback: bundled example PDBs when the source directory is absent.
if "PDB_DIR" not in os.environ and not os.path.isdir(PDB_DIR):
    _ex = os.path.join(ROOT, "examples", "pdb")
    if os.path.isdir(_ex):
        PDB_DIR = _ex


def sh(cmd, cwd, inp=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, input=inp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{r.stderr[-1500:]}")
    return r


def _elem_class(atom):
    sym = (getattr(atom, "element", None) or "")
    sym = sym.strip().upper() if sym else atom.name.strip()[:1].upper()
    return _ELEM.get(sym, 5)


def _kabsch_apply(mobile, ref, mask):
    mc, rc = mobile[mask].mean(0), ref[mask].mean(0)
    H = (mobile[mask] - mc).T @ (ref[mask] - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (mobile - mc) @ R.T + rc


def atoms_from_trajectory(gro, xtc):
    """All heavy atoms (excl. bulk water/Na/Cl) from an aligned trajectory."""
    import MDAnalysis as mda
    u = mda.Universe(gro, xtc)
    sel = u.select_atoms("not name H* and not type H "
                         "and not resname HOH SOL TIP3 SPC NA CL SOD CLA CAL MG")
    atoms = sel.atoms
    elem = np.array([_elem_class(a) for a in atoms], dtype=np.int64)
    atom_names = np.array([a.name for a in atoms])
    # residue index (compact, in topology order of first appearance)
    seen, res_map, order = {}, {}, []
    for a in atoms:
        r = int(a.resindex)
        if r not in seen:
            seen[r] = len(order); order.append(r)
    residx = np.array([seen[int(a.resindex)] for a in atoms], dtype=np.int64)
    # per-residue restype (one lookup per unique residue, via a representative atom)
    res_restype = np.full(len(order), 24, dtype=np.int64)  # 24 = other
    for a in atoms:
        ri = seen[int(a.resindex)]
        res_restype[ri] = RESTYPE.get(a.resname.strip().upper(), 24)
    coords = np.empty((len(u.trajectory), len(atoms), 3), np.float32)
    for i, _ in enumerate(u.trajectory):
        coords[i] = atoms.positions
    return coords, elem, residx, res_restype, atom_names


def process_md(pid):
    w = os.path.join(MD_DIR, pid)
    if not os.path.exists(os.path.join(w, "prod.xtc")):
        return None
    out = os.path.join(DATA, "md", f"{pid}.npz")
    if os.path.exists(out):
        return "cached"
    n = "-n index.ndx" if os.path.exists(os.path.join(w, "index.ndx")) else ""
    grp = "Protein_RNA" if n else "non-Water"
    if not os.path.exists(os.path.join(w, "solute.gro")):
        sh(f"printf '{grp}\\n' | gmx trjconv -f npt.gro -s prod.tpr {n} -o solute.gro", w)
    if not os.path.exists(os.path.join(w, "solute_pbc.xtc")):
        # -pbc cluster (NOT mol): assembles the multi-molecule complex (protein+RNA+
        # ligands) into ONE unit so they don't get imaged to opposite box sides.
        # -pbc mol images each molecule independently -> ~25% of frames had the
        # complex split across the box (corrupted training data). Needs 3 group
        # selections (cluster / center / output).
        sh(f"printf '{grp}\\n{grp}\\n{grp}\\n' | gmx trjconv -f prod.xtc -s prod.tpr {n} "
           f"-o solute_pbc.xtc -pbc cluster -center -ur compact", w)
    if not os.path.exists(os.path.join(w, "fit.xtc")):
        sh(f"printf '{grp}\\n{grp}\\n' | gmx trjconv -f solute_pbc.xtc -s prod.tpr {n} "
           f"-o fit.xtc -fit rot+trans", w)
    coords, elem, residx, res_rt, atom_names = atoms_from_trajectory(
        os.path.join(w, "solute.gro"), os.path.join(w, "fit.xtc"))
    static = coords[0].copy()
    os.makedirs(os.path.join(DATA, "md"), exist_ok=True)
    np.savez_compressed(out, coords=coords, static=static, atom_elements=elem,
                        atom_residx=residx, res_restype=res_rt, atom_names=atom_names,
                        pdb_id=pid)
    rmsd = np.sqrt(((coords - static[None]) ** 2).sum(-1).mean(-1))
    return dict(pdb_id=pid, n_frames=int(coords.shape[0]), n_atoms=int(coords.shape[1]),
                rmsd_mean_A=round(float(rmsd.mean()), 2))


def process_nmr(pid):
    out = os.path.join(DATA, "nmr", f"{pid}.npz")
    if os.path.exists(out):
        return "cached"
    d = os.path.join(PDB_DIR, pid)
    hits = ([os.path.join(d, f) for f in os.listdir(d)
             if f.lower().endswith(".pdb") and not f.startswith(".")]
            if os.path.isdir(d) else [])
    if not hits:
        return None
    # parse heavy atoms (excl. water/ions) from each MODEL
    models, cur = [], []
    atom_name_list = []
    first_model = True
    for line in open(hits[0], errors="ignore"):
        s = line[:6].strip()
        if s == "MODEL":
            cur = []
        elif s == "ENDMDL":
            if cur:
                models.append(cur)
            first_model = False
            cur = []
        elif s == "ATOM":
            name, resn = line[12:16].strip(), line[17:20].strip().upper()
            if name[:1] == "H" or resn in ("HOH", "SOL", "NA", "CL", "MG", "CA"):
                continue
            cur.append((line[21], line[22:26].strip(), resn, name[:1].upper(),
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]))
            if first_model:
                atom_name_list.append(name)
    if cur:
        models.append(cur)
    if len(models) < 2 or any(len(m) != len(models[0]) for m in models):
        return "inconsistent"
    m0 = models[0]
    elem = np.array([_ELEM.get(e, 5) for _, _, _, e, _ in m0], dtype=np.int64)
    # residue indexing
    seen, res_map = {}, []
    for ch, rs, resn, _, _ in m0:
        if (ch, rs) not in seen:
            seen[(ch, rs)] = len(res_map); res_map.append(resn)
    residx = np.array([seen[(ch, rs)] for ch, rs, _, _, _ in m0], dtype=np.int64)
    res_rt = np.array([RESTYPE.get(rn, 24) for rn in res_map], dtype=np.int64)
    atom_names = np.array(atom_name_list[:len(m0)])
    raw = np.array([[xyz for _, _, _, _, xyz in m] for m in models], dtype=np.float32)
    ca = np.array([RESTYPE.get(rn, 24) < 20 for rn in
                   [m0[i][2] for i in range(len(m0))]])  # protein atoms for alignment
    if ca.sum() >= 5:
        aligned = np.array([raw[0]] + [_kabsch_apply(m, raw[0], ca) for m in raw[1:]], dtype=np.float32)
    else:
        aligned = raw
    os.makedirs(os.path.join(DATA, "nmr"), exist_ok=True)
    np.savez_compressed(out, coords=aligned, static=aligned[0], atom_elements=elem,
                        atom_residx=residx, res_restype=res_rt, atom_names=atom_names,
                        pdb_id=pid)
    return len(models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    import pandas as pd
    man = pd.read_csv(os.path.join(DATA, "systems_manifest.csv"))
    sel = man[man.selected == 1]
    # --only may include systems NOT in the manifest (e.g. independent test set);
    # add them so they get processed even without a manifest row
    if only:
        known = set(sel.pdb_id)
        extra = only - known
        if extra:
            import pandas as _pd
            extra_rows = _pd.DataFrame({"pdb_id": sorted(extra), "selected": 1, "split": "test"})
            sel = _pd.concat([sel, extra_rows], ignore_index=True)
    qc = []
    for _, r in sel.iterrows():
        pid = r.pdb_id
        if only and pid not in only:
            continue
        try:
            res = process_md(pid)
            if isinstance(res, dict):
                res["split"] = r.split; qc.append(res)
                print(f"[md] {pid}: {res['n_frames']}f {res['n_atoms']}a")
            elif res == "cached":
                print(f"[md] {pid}: cached")
        except Exception as e:
            print(f"[md] {pid}: FAILED — {e}")
        if r.method == "NMR":
            try:
                n = process_nmr(pid)
                if isinstance(n, int):
                    print(f"[nmr] {pid}: {n} models")
            except Exception as e:
                print(f"[nmr] {pid}: FAILED — {e}")
    if qc:
        os.makedirs(RESULTS, exist_ok=True)
        pd.DataFrame(qc).to_csv(os.path.join(RESULTS, "qc_md.csv"), index=False)


if __name__ == "__main__":
    main()
