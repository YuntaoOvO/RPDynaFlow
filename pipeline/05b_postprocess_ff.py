#!/usr/bin/env python3
"""05b_postprocess_ff.py — featurize per-atom FORCES + per-frame ENERGY for the
force-matching model (Model B).

Reads the existing aligned md npz (coords/static/atom info) + the force rerun
(rerunf.trr, full system) + energy (rerunf.edr), rotates the per-atom forces into
the SAME aligned frame as the coords (Kabsch R, computed on the heavy solute), and
writes data/md_ff/<ID>.npz:
    coords (F,A,3), static (A,3), atom_elements, atom_residx, res_restype  [copied]
    forces (F,A,3)   — solute heavy-atom forces, rotated into the aligned frame
    energy (F,)      — per-frame Potential (kJ/mol, FULL system — see caveat)
    pdb_id

Caveat on energy: rerunf.edr Potential is the FULL solvated system; a solute-only
energy head can't match its absolute scale, so in flow_model_ff.py use a SMALL
lambda_e (or 0). Force-matching (per-atom, well-defined on solute) is the robust signal.

Prerequisite: pipeline/extract_forces.sh <ID>  (makes rerunf.trr + rerunf.edr).
Run:  python3 05b_postprocess_ff.py --only 1EKZ,2ESE,1NYB
"""
import argparse, os, subprocess
import numpy as np

for _g in (os.environ.get("GMXBIN", ""), "/opt/gromacs/bin", "/usr/local/gromacs/bin"):
    if _g and os.path.isdir(_g) and _g not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _g + os.pathsep + os.environ["PATH"]
os.environ["GMX_MAXBACKUP"] = "-1"

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_DIR = os.environ.get("MD_DIR", os.path.join(ROOT, "md"))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))

SEL = "not name H* and not type H and not resname HOH SOL TIP3 SPC NA CL SOD CLA CAL MG"


def kabsch_R(mobile, ref):
    """Least-squares rotation R (3x3) aligning mobile onto ref (Nx3).
    Row-vector convention: aligned = (mobile - mc) @ R.T + rc; vectors rotate as v @ R.T."""
    mc, rc = mobile.mean(0), ref.mean(0)
    H = (mobile - mc).T @ (ref - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def extract_energy(edr, w):
    """Per-frame Potential (kJ/mol) from rerunf.edr via gmx energy."""
    xvg = os.path.join(w, "_e.xvg")
    r = subprocess.run(f'printf "Potential\\n0\\n" | gmx energy -f {edr} -o {xvg}',
                       shell=True, cwd=w, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(xvg):
        raise RuntimeError(f"gmx energy failed: {(r.stderr or '')[-400:]}")
    e = np.loadtxt(xvg, comments=("#", "@"))
    os.remove(xvg)
    return e[:, 1]


def process_ff(pid):
    w = os.path.join(MD_DIR, pid)
    md_npz = os.path.join(DATA, "md", f"{pid}.npz")
    trr, edr = os.path.join(w, "rerunf.trr"), os.path.join(w, "rerunf.edr")
    need = [md_npz, trr, edr, os.path.join(w, "solute.gro"),
            os.path.join(w, "solute_pbc.xtc"), os.path.join(w, "prod.tpr")]
    miss = [os.path.basename(f) for f in need if not os.path.exists(f)]
    if miss:
        return f"missing {miss}"
    out = os.path.join(DATA, "md_ff", f"{pid}.npz")
    if os.path.exists(out):
        return "cached"

    import MDAnalysis as mda
    z = np.load(md_npz)
    static = z["static"].astype(np.float32)           # (A,3) aligned frame-0
    A = static.shape[0]

    # un-aligned heavy-solute coords (solute.gro + solute_pbc.xtc) — to derive R
    u_s = mda.Universe(os.path.join(w, "solute.gro"), os.path.join(w, "solute_pbc.xtc"))
    solute = u_s.select_atoms(SEL)
    # forces from rerun (full system) — same selection
    u_f = mda.Universe(os.path.join(w, "prod.tpr"), trr)
    fsel = u_f.select_atoms(SEL)
    nF = len(u_f.trajectory)
    if not (fsel.n_atoms == solute.n_atoms == A):
        return f"atom-count mismatch forces={fsel.n_atoms} solute={solute.n_atoms} md={A}"
    if len(u_s.trajectory) != nF:
        return f"frame-count mismatch solute={len(u_s.trajectory)} forces={nF}"

    forces = np.empty((nF, A, 3), np.float32)
    for i, ts in enumerate(u_f.trajectory):
        u_s.trajectory[i]                                 # sync solute to same frame
        R = kabsch_R(solute.positions, static)            # align this frame -> static
        forces[i] = fsel.forces @ R.T                     # rotate force vectors
    energy = extract_energy(edr, w).astype(np.float32)
    if len(energy) != nF:
        return f"energy frames {len(energy)} != {nF}"

    os.makedirs(os.path.join(DATA, "md_ff"), exist_ok=True)
    np.savez_compressed(out, coords=z["coords"], static=static,
                        atom_elements=z["atom_elements"], atom_residx=z["atom_residx"],
                        res_restype=z["res_restype"], forces=forces, energy=energy,
                        pdb_id=pid)
    f99 = float(np.percentile(np.linalg.norm(forces, axis=2), 99))
    return dict(pdb_id=pid, n_frames=nF, n_atoms=A, force_99pct_kJ=round(f99, 1),
                energy=(round(float(energy.min()), 0), round(float(energy.max()), 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    import pandas as pd
    man = pd.read_csv(os.path.join(DATA, "systems_manifest.csv"))
    only = set(args.only.split(",")) if args.only else None
    for _, r in man[man.selected == 1].iterrows():
        pid = r.pdb_id
        if only and pid not in only:
            continue
        try:
            print(f"[ff] {pid}: {process_ff(pid)}")
        except Exception as e:
            print(f"[ff] {pid}: FAILED — {e}")


if __name__ == "__main__":
    main()
