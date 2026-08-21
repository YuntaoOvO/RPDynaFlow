#!/usr/bin/env python3
"""ago2_multiconf_eval.py — BioEmu-style cross-structure coverage for Ago2 showcase.

Compares DynaFlow ensembles conditioned on 4W5N (guide-only) vs 9K6T (guide+target)
against the alternate experimental structure. Metrics follow bioemu-benchmarks multiconf
semantics: RMSD coverage curves and domain-local RMSD on sequence-aligned protein Cα.

Inputs:
  results/samples/ago2/<run>/<4W5N|9K6T>.npz  (from gen_ensembles.py --static)
  RNA-protein complexes/4W5N/*.pdb, 9K6T/*.pdb  (reference coordinates)

Run:
  python3 ago2_multiconf_eval.py --samples-dir ../results/samples/ago2/r15
  python3 ago2_multiconf_eval.py --samples-dir ../results/samples/ago2/r15 --n-gen-cap 200
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize_static_pdb import parse_static_pdb, _find_pdb
from importlib import import_module

RESTYPE = import_module("05_postprocess").RESTYPE

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
PDB_DIR = os.environ.get("PDB_DIR", os.path.join(ROOT, "RNA-protein complexes"))

# Ago2 domain boundaries (UniProt Q9UKV8, 1-indexed inclusive)
AGO2_DOMAINS = {
    "N": (1, 61),
    "L1": (62, 182),
    "PAZ": (183, 277),
    "L2": (278, 421),
    "MID": (422, 575),
    "PIWI": (576, 859),
}

COV_THRESHOLDS = np.arange(0.5, 12.01, 0.25)


def _kabsch(mobile, ref):
    mc, rc = mobile.mean(0), ref.mean(0)
    H = (mobile - mc).T @ (ref - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (mobile - mc) @ R.T + rc


def extract_ca_map(pdb_path, chain_preference=("A", "B", "C")):
    """Return dict resseq(int) -> CA xyz for protein residues on preferred chain."""
    _, _, _, res_rt, atom_names, atoms = parse_static_pdb(pdb_path)
    ca = {}
    for i, a in enumerate(atoms):
        if res_rt is not None:
            pass
        if a["name"] != "CA":
            continue
        if RESTYPE.get(a["resn"], 24) >= 20:
            continue
        try:
            rs = int(a["resseq"])
        except ValueError:
            continue
        if a["chain"] in chain_preference or len(chain_preference) == 0:
            ca[rs] = np.array(a["xyz"], dtype=np.float64)
    if not ca:
        for a in atoms:
            if a["name"] != "CA":
                continue
            if a["resn"] in ("A", "U", "G", "C"):
                continue
            try:
                rs = int(a["resseq"])
            except ValueError:
                continue
            ca[rs] = np.array(a["xyz"], dtype=np.float64)
    return ca


def common_residues(ca_a, ca_b):
    return sorted(set(ca_a.keys()) & set(ca_b.keys()))


def ca_vectors(ca_map, residues):
    return np.array([ca_map[r] for r in residues], dtype=np.float64)


def rmsd_aligned(mobile, ref):
    """mobile (F,N,3), ref (N,3) -> (F,) RMSD after per-frame Kabsch."""
    out = np.empty(len(mobile), dtype=np.float64)
    for i in range(len(mobile)):
        aligned = _kabsch(mobile[i], ref)
        out[i] = np.sqrt(((aligned - ref) ** 2).sum(-1).mean())
    return out


def coverage_at_thresholds(gen_rmsd_to_ref, thresholds=COV_THRESHOLDS, min_frac=0.001):
    """BioEmu-style: ref is a single structure; gen_rmsd_to_ref is (S,) per-sample RMSD."""
    # For single reference structure, covered if min sample RMSD < threshold
    # Extended: treat ref as one frame, coverage = fraction of ref "frames" within threshold
    # Here one ref -> covered if any sample within threshold (or k-of-n rule)
    n = len(gen_rmsd_to_ref)
    k = max(1, int(np.ceil(min_frac * n)))
    dk = np.partition(gen_rmsd_to_ref, k - 1)[k - 1]
    return np.array([float(dk < t) for t in thresholds])


def domain_masks(residues):
    """Boolean mask per domain for aligned residue list."""
    masks = {}
    for name, (lo, hi) in AGO2_DOMAINS.items():
        masks[name] = np.array([lo <= r <= hi for r in residues], dtype=bool)
    return masks


def eval_pair(gen_npz, ref_pdb, residues, label):
    z = np.load(gen_npz, allow_pickle=True)
    gen = z["gen"] if "gen" in z else z["gen_coords"]
    ref_ca = ca_vectors(extract_ca_map(ref_pdb), residues)
    pid = os.path.basename(gen_npz).replace(".npz", "")
    src = str(z["source_pdb"]) if "source_pdb" in z else ""
    if not src or not os.path.exists(src):
        src = _find_pdb(pid) or ""
    _, _, _, res_rt, _, atoms = parse_static_pdb(src)
    gen_ca_idx = [i for i, a in enumerate(atoms) if a["name"] == "CA"
                  and RESTYPE.get(a["resn"], 24) < 20
                  and int(a["resseq"]) in residues]
    if len(gen_ca_idx) != len(residues):
        # fallback: order CA atoms in file
        gen_ca_idx = [i for i, a in enumerate(atoms) if a["name"] == "CA"
                      and RESTYPE.get(a["resn"], 24) < 20][: len(residues)]
    gen_ca = gen[:, gen_ca_idx]
    rmsds = rmsd_aligned(gen_ca, ref_ca)
    cov = coverage_at_thresholds(rmsds)
    dom_masks = domain_masks(residues)
    dom_rmsd = {}
    for dname, mask in dom_masks.items():
        if mask.sum() < 5:
            continue
        sub_ref = ref_ca[mask]
        sub_gen = gen_ca[:, mask]
        dom_rmsds = rmsd_aligned(sub_gen, sub_ref)
        dom_rmsd[dname] = dict(
            mean=float(dom_rmsds.mean()),
            min=float(dom_rmsds.min()),
            p10=float(np.percentile(dom_rmsds, 10)),
        )
    i3 = int(np.where(np.isclose(COV_THRESHOLDS, 3.0))[0][0])
    i5 = int(np.where(np.isclose(COV_THRESHOLDS, 5.0))[0][0])
    return dict(
        label=label,
        n_gen=int(len(gen)),
        rmsd_mean=float(rmsds.mean()),
        rmsd_min=float(rmsds.min()),
        rmsd_p10=float(np.percentile(rmsds, 10)),
        cov_3A=float(cov[i3]),
        cov_5A=float(cov[i5]),
        cov_auc=float((getattr(np, "trapezoid", None) or np.trapz)(cov, COV_THRESHOLDS) / (COV_THRESHOLDS[-1] - COV_THRESHOLDS[0])),
        domain_rmsd=dom_rmsd,
        rmsd_per_sample=rmsds,
        coverage_curve=cov,
    )


def plot_coverage(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.plot(COV_THRESHOLDS, r["coverage_curve"], label=r["label"], lw=2)
    ax.set_xlabel("RMSD threshold (Å)")
    ax.set_ylabel("Coverage (BioEmu-style)")
    ax.set_title("Ago2 cross-structure Cα RMSD coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coverage_curves.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out", default="", help="default: results/ago2/<samples-dir basename>")
    ap.add_argument("--pdb-a", default="4W5N")
    ap.add_argument("--pdb-b", default="9K6T")
    args = ap.parse_args()

    pdb_a = _find_pdb(args.pdb_a)
    pdb_b = _find_pdb(args.pdb_b)
    if not pdb_a or not pdb_b:
        sys.exit(f"missing PDB: {args.pdb_a}={pdb_a} {args.pdb_b}={pdb_b}")

    ca_a = extract_ca_map(pdb_a)
    ca_b = extract_ca_map(pdb_b)
    residues = common_residues(ca_a, ca_b)
    print(f"aligned protein Cα residues: {len(residues)} "
          f"({residues[0]}..{residues[-1]})")

    npz_a = os.path.join(args.samples_dir, f"{args.pdb_a}.npz")
    npz_b = os.path.join(args.samples_dir, f"{args.pdb_b}.npz")
    for p in (npz_a, npz_b):
        if not os.path.exists(p):
            sys.exit(f"missing sample npz: {p}")

    results = [
        eval_pair(npz_a, pdb_b, residues,
                  f"{args.pdb_a}_gen vs {args.pdb_b}_ref"),
        eval_pair(npz_b, pdb_a, residues,
                  f"{args.pdb_b}_gen vs {args.pdb_a}_ref"),
    ]

    out_dir = args.out or os.path.join(RESULTS, "ago2", os.path.basename(args.samples_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("domain_rmsd", "rmsd_per_sample", "coverage_curve")}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "multiconf_summary.csv"), index=False)

    with open(os.path.join(out_dir, "multiconf_detail.json"), "w") as fh:
        json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in r.items()} for r in results], fh, indent=2)

    plot_coverage(results, out_dir)
    print(df.to_string(index=False))
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
