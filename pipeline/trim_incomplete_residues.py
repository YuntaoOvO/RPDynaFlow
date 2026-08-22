#!/usr/bin/env python3
"""trim_incomplete_residues.py — remove protein residues with missing heavy atoms.

A residue counts as incomplete when any atom of its minimal required set
(backbone + key side-chain heavy atoms, PDB v3 naming) is absent from ALL
altloc conformers present in the file. Incomplete residues are dropped,
together with their ANISOU/TER bookkeeping lines. Non-protein records
(RNA, ligands, ions, water) pass through unchanged.

Usage:
  python3 trim_incomplete_residues.py in.pdb [out.pdb]
  python3 trim_incomplete_residues.py in.pdb            # writes in_trimmed.pdb
"""
import os
import sys

# Minimal required heavy atoms per residue (PDB v3 / CHARMM naming).
PROTEIN_MIN = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3",
            "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}


def _res_key(line):
    """Residue identity: chain + resSeq + iCode (altloc-blank)."""
    return (line[21], line[22:27])


def trim(in_path, out_path):
    """Write out_path with incomplete protein residues removed.

    Returns (n_residues_total, n_residues_dropped, dropped_list).
    """
    with open(in_path, errors="ignore") as fh:
        lines = fh.readlines()

    # Collect the set of atom names present per residue (any altloc counts).
    present = {}
    for ln in lines:
        if ln.startswith(("ATOM  ", "HETATM")) and ln[17:20].strip() in PROTEIN_MIN:
            present.setdefault(_res_key(ln), set()).add(ln[12:16].strip())

    # Residue name is only available on each line; mark incomplete residues.
    dropped = set()
    for ln in lines:
        if ln.startswith(("ATOM  ", "HETATM")):
            resn = ln[17:20].strip()
            if resn in PROTEIN_MIN:
                k = _res_key(ln)
                if not set(PROTEIN_MIN[resn]) <= present[k]:
                    dropped.add(k)

    out, ter_pending = [], False
    for ln in lines:
        if ln.startswith("ANISOU"):
            # drop bookkeeping lines of dropped residues
            if _res_key(" " + ln[1:]) in dropped or _res_key(ln) in dropped:
                continue
        if ln.startswith(("ATOM  ", "HETATM")):
            if _res_key(ln) in dropped:
                ter_pending = True
                continue
            ter_pending = False
        elif ln.startswith("TER") and ter_pending:
            continue  # TER of a dropped residue
        out.append(ln)

    with open(out_path, "w") as fh:
        fh.writelines(out)
    return len(present), len(dropped), sorted(dropped)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.splitext(in_path)[0] + "_trimmed.pdb"
    n_total, n_drop, dropped = trim(in_path, out_path)
    print(f"{in_path}: {n_total} protein residues, dropped {n_drop} incomplete")
    for k in dropped:
        print(f"  dropped: chain {k[0].strip() or '-'} resid {k[1].strip()}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
