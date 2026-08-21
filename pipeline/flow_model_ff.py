#!/usr/bin/env python3
"""flow_model_ff.py — Model B: AtomFlowNet + CFM + energy/force head (force-matching).

Same backbone + CFM as flow_model.py (Model A), PLUS an energy head whose gradient
gives forces, trained to match MD forces (and detrended energy). Tests whether
physics supervision improves the generated ensembles.

  E(x)   = Σ_i energy_head(h_i)                 # scalar per conformation
  F_i    = −∂E/∂x_i   (autograd; x = scaled displacement delta)
  loss   = L_CFM + λ_f·‖F_pred−F_MD‖²/F_std² + λ_e·‖Ê_pred−Ê_MD‖²/Var(E_MD)

Forces are in the aligned frame (05b rotates them by the same Kabsch R as coords).
delta=(coords−static)/SCALE ⇒ physical force = (−∇_delta E)/SCALE.
Energy term is detrended per-batch (E_MD is full-system; only the relative landscape
is meaningful for a solute head) — keep λ_e small. Single continuous run (no chunking).

Run:  python3 flow_model_ff.py --epochs 300 --systems 1EKZ,2ESE,1NYB
"""
import argparse, os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from flow_model import (AtomFlowNet, dist_rbf, build_graph, _to_dev, sample_ensemble,
                        N_ELEM, N_TYPES, SCALE, CUTOFF, DATA, RESULTS)

FF_DIR = os.path.join(RESULTS, "ff")


class AtomFlowNetFF(AtomFlowNet):
    """AtomFlowNet + per-atom energy head (summed to a scalar E)."""
    def __init__(self, d=128, n_layers=5):
        super().__init__(d, n_layers)
        self.energy_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def energy(self, x, static, elem, residx, restype, ei_j, ei_i, edge_rbf, t):
        h = self.features(x, static, elem, residx, restype, ei_j, ei_i, edge_rbf, t)
        return self.energy_head(h).sum(dim=1)            # (B,)


def load_systems_ff(pids):
    """load_systems variant that also carries forces (F,A,3, physical) + energy (F,)."""
    out = []
    for pid in pids:
        p = os.path.join(DATA, "md_ff", f"{pid}.npz")
        if not os.path.exists(p):
            print(f"[ff] {pid}: no md_ff npz — skip"); continue
        z = np.load(p)
        coords, static = z["coords"].astype(np.float32), z["static"].astype(np.float32)
        ei_i, ei_j, edge_dist = build_graph(static)
        out.append(dict(delta=(coords - static[None]) / SCALE,
                        static=(static / SCALE).astype(np.float32),
                        forces=z["forces"].astype(np.float32),     # physical kJ/mol/Å (aligned)
                        energy=z["energy"].astype(np.float32),     # full-system Potential
                        elem=z["atom_elements"].astype(np.int64),
                        residx=z["atom_residx"].astype(np.int64),
                        restype=z["res_restype"].astype(np.int64),
                        edge_rbf=dist_rbf(edge_dist), ei_i=ei_i, ei_j=ei_j,
                        pdb_id=str(pid), F=coords.shape[0], A=coords.shape[1]))
    return out


def sample_ensemble_guided(model, sysd, n_samples=50, steps=30, sigma=1.0, eta=0.1,
                           device="cuda", chunk=5):
    """Energy-guided CFM sampling (Model C): each Euler velocity step is followed by a
    nudge down the learned energy E (unit-gradient step of size `eta`, ~eta*SCALE A).
    Needs an energy head (AtomFlowNetFF / B's weights). -> absolute atom coords (S,A,3) A.
    `eta` is the guidance strength — sweep it at benchmark time (too high distorts the
    CFM trajectory, too low ≈ plain sampling)."""
    model.eval()
    outs = []
    st = sysd["static"]
    for s in range(0, n_samples, chunk):
        b = min(chunk, n_samples - s)
        x = torch.randn(b, sysd["A"], 3, device=device) * sigma
        for k in range(steps):
            t = torch.full((b,), k / steps, device=device)
            with torch.no_grad():
                v = model(x, st.expand(b, -1, -1), sysd["elem"].expand(b, -1),
                          sysd["residx"].expand(b, -1), sysd["restype"], sysd["ei_j"],
                          sysd["ei_i"], sysd["edge_rbf"], t)
                x = x + v / steps
            x = x.detach().requires_grad_(True)
            E = model.energy(x, st.expand(b, -1, -1), sysd["elem"].expand(b, -1),
                             sysd["residx"].expand(b, -1), sysd["restype"], sysd["ei_j"],
                             sysd["ei_i"], sysd["edge_rbf"], t)
            g = torch.autograd.grad(E.sum(), x)[0]
            norm = g.reshape(b, -1).norm(dim=1).clamp(min=1e-8)      # (b,)
            step = eta * g / norm[:, None, None]                     # (b,1,1) broadcasts with (b,A,3)
            x = (x - step).detach()
        outs.append((st.expand(b, -1, -1) + x).cpu().numpy() * SCALE)
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--systems", default="1EKZ,2ESE,1NYB")
    ap.add_argument("--frames-per-step", type=int, default=8)
    ap.add_argument("--ff-frames", type=int, default=2,
                    help="subset of frames-per-step used for force/energy autograd (memory control)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-f", type=float, default=1.0)
    ap.add_argument("--lambda-e", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="",
                    help="output checkpoint path (else results/ff/flow_model_ff.pt)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pids = [s.strip().upper() for s in args.systems.split(",") if s.strip()]
    train = load_systems_ff(pids)
    assert train, "no md_ff npz — run extract_forces.sh + 05b_postprocess_ff.py first"
    train = [_to_dev(t, device) for t in train]

    # scales (computed once)
    sigma = float(np.concatenate([t["delta"].reshape(-1, 3) for t in train]).std())
    Fstd = float(np.concatenate([t["forces"].reshape(-1, 3) for t in train]).std())
    Econ = np.concatenate([t["energy"] for t in train])
    Evar = float(Econ.var()) or 1.0
    print(f"[ff] train {[t['pdb_id'] for t in train]} | sigma={sigma:.3f} "
          f"F_std={Fstd:.1f} kJ/mol/Å | E_var={Evar:.2e} | λ_f={args.lambda_f} λ_e={args.lambda_e}")

    model = AtomFlowNetFF().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    K = args.frames_per_step

    for ep in tqdm(range(args.epochs), desc="[B]ff", unit="ep"):
        model.train(); lc = lf = le = 0.0; nb = 0
        # shuffled full-coverage epoch: every frame seen exactly once
        all_batches = []
        for si, sysd in enumerate(train):
            idx = np.random.permutation(sysd["F"])
            for i in range(0, sysd["F"], K):
                all_batches.append((si, idx[i:i + K]))
        np.random.shuffle(all_batches)
        for si, fi in all_batches:
            sysd = train[si]
            x1 = torch.from_numpy(sysd["delta"][fi]).to(device)          # (K,A,3) scaled
            F_MD = torch.from_numpy(sysd["forces"][fi]).to(device)        # (K,A,3) physical
            E_MD = torch.from_numpy(sysd["energy"][fi]).to(device)        # (K,)
            B, A, _ = x1.shape
            st = sysd["static"].expand(B, -1, -1)

            # --- CFM loss (at x_t) ---
            z = torch.randn_like(x1) * sigma
            t = torch.rand(B, device=device); tm = t[:, None, None]
            x_t = (1 - tm) * z + tm * x1
            v = model(x_t, st, sysd["elem"].expand(B, -1), sysd["residx"].expand(B, -1),
                      sysd["restype"], sysd["ei_j"], sysd["ei_i"], sysd["edge_rbf"], t)
            loss_cfm = ((v - (x1 - z)) ** 2).mean()

            # --- force + energy loss (at real displacement x1, autograd) ---
            # use only ff_frames subset for autograd to control memory (create_graph=True
            # on 8×2099 atoms OOMs a 32 GB card; 2 frames is safe and still provides
            # physical supervision signal)
            nf = min(args.ff_frames, B)
            x1g = x1[:nf].detach().clone().requires_grad_(True)
            st_g = st[:nf]
            E = model.energy(x1g, st_g, sysd["elem"].expand(nf, -1), sysd["residx"].expand(nf, -1),
                             sysd["restype"], sysd["ei_j"], sysd["ei_i"], sysd["edge_rbf"], t[:nf])
            grad = torch.autograd.grad(E.sum(), x1g, create_graph=True)[0]
            F_pred = -grad / SCALE                                      # physical kJ/mol/Å
            loss_f = ((F_pred - F_MD[:nf]) ** 2).mean() / (Fstd ** 2)
            Ed, ED = E - E.mean(), E_MD[:nf] - E_MD[:nf].mean()
            loss_e = (Ed - ED).pow(2).mean() / Evar

            loss = loss_cfm + args.lambda_f * loss_f + args.lambda_e * loss_e
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            lc += loss_cfm.item(); lf += loss_f.item(); le += loss_e.item(); nb += 1
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[ff] epoch {ep+1:4d}/{args.epochs}  cfm {lc/nb:.4f}  "
                  f"force {lf/nb:.3f}  energy {le/nb:.3f}")

    os.makedirs(FF_DIR, exist_ok=True)
    path = args.save if args.save else os.path.join(FF_DIR, "flow_model_ff.pt")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), sigma=sigma, F_std=Fstd, E_var=Evar,
                    config="atom-gnn-cfm-ff", systems=[t["pdb_id"] for t in train],
                    lambda_f=args.lambda_f, lambda_e=args.lambda_e), path)
    print(f"[ff] saved {path}")


if __name__ == "__main__":
    main()
