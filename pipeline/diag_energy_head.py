#!/usr/bin/env python3
"""diag_energy_head.py — diagnose a trained B (AtomFlowNetFF) energy head.

Answers two questions the training loss cannot:
  1. Did the energy head learn anything? (energy loss ~ 0 is the trivial floor,
     not evidence either way) -> Pearson corr between E_pred and E_MD across
     frames of the SAME system, at t=1 and t=0.5.
  2. Where does the force loss (stuck ~0.7) live? -> per-element and
     per-residue-class R^2 of F_pred vs F_MD.

Run on GPU (CPU inference hangs):
  python3 diag_energy_head.py --ckpt ../results/ff/flow_model_ff.pt --systems 1NYB,2ESE,1EKZ
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from flow_model import _to_dev, SCALE
from flow_model_ff import AtomFlowNetFF, load_systems_ff

ELEM_NAMES = ["C", "N", "O", "P", "S", "other"]
RES_NAMES = ["protein", "RNA", "other"]


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def r2(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--systems", default="1NYB,2ESE,1EKZ")
    ap.add_argument("--frames", type=int, default=64, help="frames for energy correlation")
    ap.add_argument("--ff-frames", type=int, default=8, help="frames for force decomposition")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "run on GPU (CPU inference hangs)"
    device = "cuda"

    ckpt = torch.load(args.ckpt, map_location=device)
    model = AtomFlowNetFF().to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    Fstd = ckpt.get("F_std", float("nan"))
    print(f"ckpt={os.path.basename(args.ckpt)} trained_on={ckpt.get('systems', [])} F_std={Fstd}")

    pids = [s.strip().upper() for s in args.systems.split(",") if s.strip()]
    for sysd_np in load_systems_ff(pids):
        pid = sysd_np["pdb_id"]
        sysd = _to_dev(sysd_np, device)
        F = sysd_np["F"]

        # --- energy correlation (no grad) ---
        fi = np.linspace(0, F - 1, min(args.frames, F)).astype(int)
        x1 = torch.from_numpy(sysd_np["delta"][fi]).to(device)
        E_MD = sysd_np["energy"][fi]
        B = len(fi)
        st = sysd["static"].expand(B, -1, -1)
        kw = dict(elem=sysd["elem"].expand(B, -1), residx=sysd["residx"].expand(B, -1),
                  restype=sysd["restype"], ei_j=sysd["ei_j"], ei_i=sysd["ei_i"],
                  edge_rbf=sysd["edge_rbf"])
        with torch.no_grad():
            E_t1 = model.energy(x1, st, kw["elem"], kw["residx"], kw["restype"],
                                kw["ei_j"], kw["ei_i"], kw["edge_rbf"],
                                torch.ones(B, device=device)).cpu().numpy()
            E_t5 = model.energy(x1, st, kw["elem"], kw["residx"], kw["restype"],
                                kw["ei_j"], kw["ei_i"], kw["edge_rbf"],
                                torch.full((B,), 0.5, device=device)).cpu().numpy()
        c1, c5 = pearson(E_t1, E_MD), pearson(E_t5, E_MD)
        print(f"\n{pid} (A={sysd_np['A']}, F={F})")
        print(f"  corr(E_pred, E_MD) within-system: t=1.0 {c1:+.3f} | t=0.5 {c5:+.3f}"
              f"  {'<- head is trivial' if max(abs(c1), abs(c5)) < 0.1 else ''}")

        # --- force decomposition (autograd, few frames) ---
        nf = min(args.ff_frames, F)
        fi2 = np.linspace(0, F - 1, nf).astype(int)
        x1g = torch.from_numpy(sysd_np["delta"][fi2]).to(device).requires_grad_(True)
        st2 = sysd["static"].expand(nf, -1, -1)
        E = model.energy(x1g, st2, sysd["elem"].expand(nf, -1),
                         sysd["residx"].expand(nf, -1), sysd["restype"],
                         sysd["ei_j"], sysd["ei_i"], sysd["edge_rbf"],
                         torch.ones(nf, device=device))
        F_pred = (-torch.autograd.grad(E.sum(), x1g)[0] / SCALE).cpu().numpy()
        F_MD = sysd_np["forces"][fi2]
        print(f"  force R^2 overall: {r2(F_pred, F_MD):+.3f}")
        elem = sysd_np["elem"]; restype = sysd_np["restype"]; residx = sysd_np["residx"]
        F_MD_by = F_MD.reshape(nf, -1, 3)
        F_pred_by = F_pred.reshape(nf, -1, 3)
        by_elem = " ".join(f"{ELEM_NAMES[e]}={r2(F_pred_by[:, elem == e], F_MD_by[:, elem == e]):+.2f}"
                           for e in range(6) if (elem == e).any())
        print(f"    by element: {by_elem}")
        rcls = np.where(restype < 20, 0, np.where(restype < 24, 1, 2))[residx]
        by_res = " ".join(f"{RES_NAMES[c]}={r2(F_pred_by[:, rcls == c], F_MD_by[:, rcls == c]):+.2f}"
                          for c in range(3) if (rcls == c).any())
        print(f"    by resclass: {by_res}")


if __name__ == "__main__":
    main()
