#!/usr/bin/env python3
"""inference_demo.py — quick inference sanity check (de-risk before long training).

Loads a checkpoint, samples an ensemble for one system, writes a viewable PDB, and
prints RMSF vs MD/NMR. Catches major issues (NaN, shape errors, atom mismatch, garbage
structures) early. RUN ON GPU — CPU inference of the GNN hangs (D-state, unkillable):
    python3 inference_demo.py [ckpt] [pdb_id] [n_gen]
defaults: ckpt=<repo>/checkpoints/flow_model_r15.pt, 4W5N, 30.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from flow_model import (AtomFlowNet, load_systems, sample_ensemble, _to_dev,
                        SCALE, DATA, RESULTS)

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MD_DIR = os.environ.get("MD_DIR", os.path.join(ROOT, "md"))


def rmsf(c):
    return np.sqrt(((c - c.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))

def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def write_pdb(gen, gro, out):
    import MDAnalysis as mda
    u = mda.Universe(gro)
    atoms = u.select_atoms("not name H* and not resname HOH SOL NA CL")
    assert atoms.n_atoms == gen.shape[1], f"atom mismatch: gro {atoms.n_atoms} vs gen {gen.shape[1]}"
    with mda.Writer(out, atoms.n_atoms) as w:
        for i in range(gen.shape[0]):
            atoms.positions = gen[i]
            w.write(atoms)


def main():
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _shipped = os.path.join(_repo, "checkpoints", "flow_model_r15.pt")
    _trained = os.path.join(RESULTS, "checkpoints", "flow_model_r15.pt")
    _default = _shipped if os.path.isfile(_shipped) else _trained
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else _default
    pid = (sys.argv[2] if len(sys.argv) > 2 else "4W5N").upper()
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | ckpt={os.path.basename(ckpt_path)} | system={pid} | n_gen={n_gen}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "ff" in ckpt.get("config", ""):
        from flow_model_ff import AtomFlowNetFF
        model = AtomFlowNetFF().to(device)
    else:
        model = AtomFlowNet().to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    sigma = ckpt["sigma"]
    print(f"config={ckpt.get('config','?')} sigma={sigma:.3f} trained_on={ckpt.get('systems',[])}")

    sysd = _to_dev(load_systems([pid], static=True)[0], device)
    static_abs = sysd["static"].cpu().numpy() * SCALE

    gen = sample_ensemble(model, sysd, n_samples=n_gen, sigma=sigma, device=device)
    finite = bool(np.isfinite(gen).all())
    rmsd_gen = np.sqrt(((gen - static_abs[None]) ** 2).sum(-1).mean(-1))
    print(f"sampled {gen.shape} | finite={finite} | gen RMSD-to-static: "
          f"mean={rmsd_gen.mean():.2f} max={rmsd_gen.max():.2f} A")

    out = os.path.join(RESULTS, f"gen_{pid}.pdb")
    gro = os.path.join(MD_DIR, pid, "solute.gro")
    if os.path.exists(gro):
        md = (sysd["delta"] + sysd["static"].cpu().numpy()) * SCALE
        r_md, r_gen = rmsf(md), rmsf(gen)
        print(f"RMSF pearson vs MD : {pearson(r_md, r_gen):.3f} | "
              f"mean rmsf md={r_md.mean():.2f} gen={r_gen.mean():.2f}")
        nmr_p = os.path.join(DATA, "nmr", f"{pid}.npz")
        if os.path.exists(nmr_p):
            r_nmr = rmsf(np.load(nmr_p)["coords"])
            print(f"RMSF pearson vs NMR: {pearson(r_nmr, r_gen):.3f}")
        write_pdb(gen, gro, out)
        print(f"wrote {out} ({gen.shape[0]} conformations) — load in PyMOL with solute.gro")
    else:
        # static-only system (zero-shot): use the source PDB as template
        from gen_ensembles import write_pdb_multimodel, _find_template_pdb
        tpl = _find_template_pdb(pid)
        write_pdb_multimodel(gen, tpl, out)
        print(f"wrote {out} ({gen.shape[0]} conformations, template: {tpl})")


if __name__ == "__main__":
    main()
