#!/usr/bin/env python3
"""drpscore_prep.py — generate DRPScore-ready interface PDBs from npz frames.

Replaces DRPScore's fragile 6A_calculation/rpo.py with a vectorized equivalent:
per frame, protein residues (20 standard aa) get chain A, RNA residues
(A/U/G/C) get chain B, and the interface = residues with any atom within
6 Angstrom of the other molecule (KDTree, per frame — the interface can
rearrange across the ensemble). Ligands/ions are dropped because
DRPScore's ModifyName.py exits on non-canonical residue names.
Output PDB columns follow the bundled Acomplex*.pdb convention so
data/predict/Main.py (BioPython) parses them directly.

Run:  python3 drpscore_prep.py --samples ../results/samples/flow_model_r15 --tag A_r15
      python3 drpscore_prep.py --md --systems 2LBS,6TPH,2HGH,7K9D,4M4O --tag MD
"""
import argparse, os
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_DIR = os.environ.get("MD_DIR", os.path.join(ROOT, "md"))

# identical to 05_postprocess.py selection
SEL = ("not name H* and not type H "
       "and not resname HOH SOL TIP3 SPC NA CL SOD CLA CAL MG")

PROT_RES = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL"}
RNA_RES = {"A", "U", "G", "C"}

CUTOFF = 6.0

# CHARMM36 (GROMACS gro) -> PDB-standard atom names, so DRPScore's
# charge dictionaries (keyed on PDB names) hit the right branch.
# ILE's terminal methyl is 'CD' in charmm36, 'CD1' in PDB.
RENAME_ATOM = {("ILE", "CD"): "CD1"}

PDB_FMT = "ATOM  {serial:5d} {name:<4s} {resname:>3s} {chain}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}\n"


def load_template(gro_path):
    import MDAnalysis as mda
    u = mda.Universe(gro_path)
    atoms = u.select_atoms(SEL)
    resnames = atoms.residues.resnames
    prot_res_mask = np.array([r in PROT_RES for r in resnames])
    rna_res_mask = np.array([r in RNA_RES for r in resnames])
    return atoms, prot_res_mask, rna_res_mask


def frame_interface_pdb(coords, atoms, prot_res_mask, rna_res_mask, out_path):
    """coords (A,3) for one frame -> interface-only PDB, protein=A, RNA=B."""
    resids = atoms.residues.resids
    resnames = atoms.residues.resnames
    # atom -> residue index map
    atom_residx = atoms.resindices
    prot_atoms = prot_res_mask[atom_residx]
    rna_atoms = rna_res_mask[atom_residx]

    pc, rc = coords[prot_atoms], coords[rna_atoms]
    tree = cKDTree(rc)
    # which protein atoms have any RNA atom within CUTOFF, and vice versa
    near_prot = np.array([len(h) > 0 for h in tree.query_ball_point(pc, CUTOFF)])
    tree_p = cKDTree(pc)
    near_rna = np.array([len(h) > 0 for h in tree_p.query_ball_point(rc, CUTOFF)])

    # interface residue ids
    prot_iface = set(atom_residx[prot_atoms][near_prot].tolist())
    rna_iface = set(atom_residx[rna_atoms][near_rna].tolist())

    serial = 0
    lines = []
    for rix in sorted(prot_iface):
        m = atom_residx == rix
        for ai in np.where(m)[0]:
            serial += 1
            nm = RENAME_ATOM.get((resnames[rix], atoms.names[ai]), atoms.names[ai])
            lines.append(PDB_FMT.format(
                serial=serial, name=nm, resname=resnames[rix],
                chain="A", resid=resids[rix],
                x=coords[ai, 0], y=coords[ai, 1], z=coords[ai, 2],
                occ=1.00, bfac=0.00))
    lines.append("TER\n")
    for rix in sorted(rna_iface):
        m = atom_residx == rix
        for ai in np.where(m)[0]:
            serial += 1
            nm = RENAME_ATOM.get((resnames[rix], atoms.names[ai]), atoms.names[ai])
            lines.append(PDB_FMT.format(
                serial=serial, name=nm, resname=resnames[rix],
                chain="B", resid=resids[rix],
                x=coords[ai, 0], y=coords[ai, 1], z=coords[ai, 2],
                occ=1.00, bfac=0.00))
    lines.append("TER\nEND\n")
    with open(out_path, "w") as f:
        f.writelines(lines)
    return len(prot_iface) + len(rna_iface)


def process(coords, pid, atoms, prot_mask, rna_mask, outdir, pdblist, max_frames):
    n = coords.shape[0]
    if max_frames and n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).round().astype(int)
    else:
        idx = np.arange(n)
    os.makedirs(outdir, exist_ok=True)
    nres = []
    for k, fi in enumerate(idx):
        out_p = os.path.join(outdir, f"{pid}_f{k:04d}.pdb")
        nres.append(frame_interface_pdb(coords[fi], atoms, prot_mask, rna_mask, out_p))
        pdblist.append(out_p)
    return len(idx), int(np.mean(nres))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="", help="dir with <ID>.npz sample files")
    ap.add_argument("--md", action="store_true", help="use data/md npz (MD frames)")
    ap.add_argument("--systems", default="", help="comma-sep PDB IDs (else all in dir)")
    ap.add_argument("--tag", required=True, help="output tag, e.g. A_r15 or MD")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per system")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "drpscore"))
    args = ap.parse_args()

    if args.md:
        src_dir, key = os.path.join(ROOT, "data", "md"), "coords"
    else:
        src_dir, key = args.samples, "gen_coords"

    pids = ([s.strip().upper() for s in args.systems.split(",") if s.strip()]
            if args.systems else
            sorted(f.replace(".npz", "").upper()
                   for f in os.listdir(src_dir) if f.endswith(".npz")))

    outdir = os.path.join(args.out, f"pdb_{args.tag}")
    pdblist = []
    for pid in pids:
        npz_p = os.path.join(src_dir, f"{pid}.npz")
        gro_p = os.path.join(MD_DIR, pid, "solute.gro")
        if not os.path.exists(npz_p):
            print(f"[skip] {pid}: no {npz_p}"); continue
        if not os.path.exists(gro_p):
            print(f"[skip] {pid}: no {gro_p}"); continue
        atoms, prot_mask, rna_mask = load_template(gro_p)
        coords = np.load(npz_p)[key].astype(np.float64)
        assert atoms.n_atoms == coords.shape[1], \
            f"{pid}: gro {atoms.n_atoms} vs npz {coords.shape[1]}"
        nf, mean_res = process(coords, pid, atoms, prot_mask, rna_mask,
                               outdir, pdblist, args.max_frames)
        print(f"[{args.tag}] {pid}: {nf} frames, mean interface residues {mean_res}")

    list_p = os.path.join(args.out, f"pdblist_{args.tag}.txt")
    with open(list_p, "w") as f:
        f.write("\n".join(pdblist) + "\n")
    print(f"[{args.tag}] {len(pdblist)} PDBs, list -> {list_p}")


if __name__ == "__main__":
    main()
