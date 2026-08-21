#!/usr/bin/env python3
"""gen_ensembles.py — sample ensembles for several systems and dump them for
download / offline analysis. GPU-only (CPU inference of the GNN hangs).

Writes, per system, into --out:
  <PDBID>.npz  gen (S,A,3) + static + atom metadata + provenance
  <PDBID>.pdb  multi-MODEL viewable ensemble (needs md/<ID>/solute.gro)
and a small manifest.csv with quick sanity numbers (RMSF pearson, amplitude,
bond-length sanity) so a glance tells you whether the samples are usable.

This is deliberately separate from 07_benchmark.py: no metrics beyond a sanity
glance, no figures — its job is to produce artifacts for 09_dynbench and for the
user to look at in PyMOL.

Run:  python3 gen_ensembles.py --ckpt results/checkpoints/flow_model_r7.pt \
          --systems 6GBM,2FY1,1A1T --n-gen 200 --out results/samples/r7
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from flow_model import (AtomFlowNet, sample_ensemble, load_systems, load_split,
                        _to_dev, DATA, RESULTS, SCALE)

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_DIR = os.environ.get("MD_DIR", os.path.join(ROOT, "md"))
PDB_DIR = os.environ.get("PDB_DIR", os.path.join(ROOT, "RNA-protein complexes"))
# Fresh-clone fallback: bundled example PDBs when the source directory is absent.
if "PDB_DIR" not in os.environ and not os.path.isdir(PDB_DIR):
    _ex = os.path.join(ROOT, "examples", "pdb")
    if os.path.isdir(_ex):
        PDB_DIR = _ex


def rmsf(c):
    return np.sqrt(((c - c.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def bonded_sanity(coords, static, cutoff=1.8):
    """Cheap covalent-geometry check without a topology: treat static heavy-atom
    pairs closer than `cutoff` A as bonds, then report how much those distances
    drift in the generated frames. Real heavy-atom bonds are 1.2-1.6 A and stiff,
    so a large mean |dev| means visibly broken chemistry.
    Returns (n_bonds, mean_abs_dev_A, frac_dev_gt_0.3A)."""
    d = np.linalg.norm(static[:, None, :] - static[None, :, :], axis=-1)
    iu = np.triu_indices(len(static), k=1)
    m = d[iu] < cutoff
    i, j = iu[0][m], iu[1][m]
    if len(i) == 0:
        return 0, float("nan"), float("nan")
    ref = d[i, j]
    gen_d = np.linalg.norm(coords[:, i, :] - coords[:, j, :], axis=-1)   # (S,nb)
    dev = np.abs(gen_d - ref[None])
    return int(len(i)), float(dev.mean()), float((dev > 0.3).mean())


def write_pdb(gen, gro, out):
    import MDAnalysis as mda
    u = mda.Universe(gro)
    atoms = u.select_atoms("not name H* and not resname HOH SOL NA CL")
    if atoms.n_atoms != gen.shape[1]:
        raise AssertionError(f"atom mismatch: gro {atoms.n_atoms} vs gen {gen.shape[1]}")
    with mda.Writer(out, atoms.n_atoms) as w:
        for i in range(gen.shape[0]):
            atoms.positions = gen[i]
            w.write(atoms)


def _find_template_pdb(pid, source_pdb=""):
    if source_pdb and os.path.exists(source_pdb):
        return source_pdb
    d = os.path.join(PDB_DIR, pid)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".pdb") and not f.startswith("."):
                return os.path.join(d, f)
    for p in (os.path.join(PDB_DIR, f"{pid.lower()}.pdb"),
              os.path.join(PDB_DIR, f"{pid.upper()}.pdb")):
        if os.path.exists(p):
            return p
    return None


def write_pdb_multimodel(gen, template_pdb, out, max_models=50):
    """Write a multi-MODEL PDB by updating coordinates from featurize_static_pdb order."""
    from featurize_static_pdb import parse_static_pdb
    _, _, _, _, _, records = parse_static_pdb(template_pdb)
    if len(records) != gen.shape[1]:
        raise AssertionError(f"atom mismatch: template {len(records)} vs gen {gen.shape[1]}")
    n_models = min(max_models, gen.shape[0])
    with open(out, "w") as fh:
        for mi in range(n_models):
            fh.write(f"MODEL     {mi + 1:4d}\n")
            for ai, rec in enumerate(records):
                x, y, z = gen[mi, ai]
                line = rec["line"]
                fh.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}\n")
            fh.write("ENDMDL\n")
        fh.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--systems", default="", help="comma-sep pdb_ids (else split=test)")
    ap.add_argument("--n-gen", type=int, default=200)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--guided", type=float, default=0.0, help="eta for Model C guided sampling")
    ap.add_argument("--static", action="store_true",
                    help="load from data/static/ (PDB-only, no MD reference)")
    ap.add_argument("--no-pdb", action="store_true", help="skip the viewable PDB")
    ap.add_argument("--chunk-size", type=int, default=20,
                    help="batch size for sampling (default: 20, auto-scales if chunk_auto enabled)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        sys.exit("refusing to run on CPU — GNN inference hangs (D-state). Need a GPU.")
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", "")
    if "ff" in cfg:
        from flow_model_ff import AtomFlowNetFF
        model = AtomFlowNetFF().to(device)
    else:
        model = AtomFlowNet().to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    sigma = ckpt["sigma"]
    trained_on = list(ckpt.get("systems", []))
    print(f"ckpt={args.ckpt} config={cfg or '?'} sigma={sigma:.3f} n_gen={args.n_gen} "
          f"guided={args.guided}\ntrained_on={trained_on}")

    if args.systems:
        pids = [s.strip().upper() for s in args.systems.split(",") if s.strip()]
        systems = load_systems(pids, static=args.static)
    else:
        systems = load_split("test") if not args.static else []

    rows = []
    for sysd in systems:
        pid = sysd["pdb_id"]
        sysd = _to_dev(sysd, device)
        static_abs = sysd["static"].cpu().numpy() * SCALE
        is_static = sysd.get("source") == "static"
        md = None if is_static else (sysd["delta"] * SCALE) + static_abs[None]
        try:
            if args.guided > 0:
                from flow_model_ff import sample_ensemble_guided
                gen = sample_ensemble_guided(model, sysd, n_samples=args.n_gen,
                                             steps=args.steps, sigma=sigma,
                                             eta=args.guided, device=device)
            else:
                gen = sample_ensemble(model, sysd, n_samples=args.n_gen,
                                      steps=args.steps, sigma=sigma, device=device,
                                      chunk=args.chunk_size)
        except Exception as e:
            print(f"[fail] {pid}: {e}")
            continue

        finite = bool(np.isfinite(gen).all())
        r_gen = rmsf(gen)
        r_md = rmsf(md) if md is not None else None
        rmsd_gen = np.sqrt(((gen - static_abs[None]) ** 2).sum(-1).mean(-1))
        rmsd_md = (np.sqrt(((md - static_abs[None]) ** 2).sum(-1).mean(-1))
                   if md is not None else None)
        nb, bdev, bfrac = bonded_sanity(gen, static_abs)

        npz = os.path.join(args.out, f"{pid}.npz")
        np.savez_compressed(
            npz, gen=gen.astype(np.float32), static=static_abs.astype(np.float32),
            atom_elements=sysd["elem"].cpu().numpy().astype(np.int64),
            atom_residx=sysd["residx"].cpu().numpy().astype(np.int64),
            res_restype=sysd["restype"].cpu().numpy().astype(np.int64),
            ckpt=os.path.basename(args.ckpt), guided_eta=float(args.guided),
            sigma=float(sigma), n_gen=int(gen.shape[0]), steps=int(args.steps),
            trained_on=np.array(trained_on, dtype=object), in_train=pid in trained_on,
            source_pdb=sysd.get("source_pdb", _find_template_pdb(pid) or ""))

        pdb_note = ""
        if not args.no_pdb:
            pdb_out = os.path.join(args.out, f"{pid}.pdb")
            try:
                if is_static:
                    tpl = _find_template_pdb(pid, sysd.get("source_pdb", ""))
                    write_pdb_multimodel(gen, tpl, pdb_out)
                else:
                    gro = os.path.join(MD_DIR, pid, "solute.gro")
                    write_pdb(gen[:min(50, len(gen))], gro, pdb_out)
                pdb_note = " +pdb"
            except Exception as e:
                pdb_note = f" (pdb skipped: {e})"

        row = dict(pdb_id=pid, in_train=pid in trained_on, static_only=is_static,
                   n_atoms=int(gen.shape[1]), n_gen=int(gen.shape[0]), finite=finite,
                   rmsf_pearson=round(pearson(r_md, r_gen), 3) if r_md is not None else None,
                   mean_rmsf_md=round(float(r_md.mean()), 3) if r_md is not None else None,
                   mean_rmsf_gen=round(float(r_gen.mean()), 3),
                   rmsd_md_mean=round(float(rmsd_md.mean()), 2) if rmsd_md is not None else None,
                   rmsd_gen_mean=round(float(rmsd_gen.mean()), 2),
                   n_bonds=nb, bond_dev_A=round(bdev, 3), bond_bad_frac=round(bfrac, 3))
        rows.append(row)
        r_str = row['rmsf_pearson'] if row['rmsf_pearson'] is not None else "n/a"
        md_rmsf = row['mean_rmsf_md'] if row['mean_rmsf_md'] is not None else "n/a"
        md_rmsd = row['rmsd_md_mean'] if row['rmsd_md_mean'] is not None else "n/a"
        print(f"{pid}: r={r_str} rmsf md/gen={md_rmsf}/"
              f"{row['mean_rmsf_gen']} rmsd md/gen={md_rmsd}/{row['rmsd_gen_mean']} "
              f"bond_dev={row['bond_dev_A']}A bad={row['bond_bad_frac']}{pdb_note}")

    if not rows:
        print("nothing generated"); return
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {len(rows)} ensembles -> {args.out}")


if __name__ == "__main__":
    main()
