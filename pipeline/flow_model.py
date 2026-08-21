#!/usr/bin/env python3
"""flow_model.py — atomic conditional flow matching for protein-RNA conformational
ensembles, with a geometry-aware GNN (SchNet-style interaction blocks).

All PBC / alignment is done upstream by gmx trjconv (see 05_postprocess.py); this
file reads clean aligned atomic coordinates and learns the per-atom displacement
ensemble around the static structure.

  nodes  = atoms (heavy atoms, protein + RNA + ligands)
  edges  = pairs within `cutoff` on the STATIC structure (fixed graph per system);
           edge feature = Gaussian RBF of the static distance (geometry-aware)
  state  = per-atom Cartesian displacement x1 = (coords - static)/SCALE
  CFM    = x_t = (1-t) z + t x1 , v* = x1 - z , z ~ N(0, sigma^2)
  head   = per-atom velocity (3) in the aligned frame

Correct & tractable: sparse graph (not full A×A), invariant distance features,
Cartesian output (valid because gmx fixes a canonical aligned frame).
"""
import argparse, math, os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
N_ELEM = 6            # C,N,O,P,S,other
N_TYPES = 24          # 20 aa + 4 nt
SCALE = 10.0          # units of 10 A
CUTOFF = 8.0          # neighbor cutoff (A) on the static structure
N_RBF = 16


def dist_rbf(d, n=N_RBF, cutoff=CUTOFF):
    """Gaussian RBF of distance d (...,) -> (...,n), cutoff-smoothed."""
    centers = torch.linspace(0, cutoff, n, device=d.device)
    gamma = 1.0 / (cutoff / n) ** 2
    rbf = torch.exp(-gamma * (d[..., None] - centers) ** 2)
    smooth = 0.5 * (torch.cos(math.pi * d / cutoff) + 1)  # envelope (0 at cutoff)
    return rbf * smooth[..., None]


class InteractionBlock(nn.Module):
    """SchNet-style: h_i += sum_{j in N(i)} filter(d_ij) ⊙ (mlp(h_j))."""
    def __init__(self, d):
        super().__init__()
        self.filt = nn.Sequential(nn.Linear(N_RBF, d), nn.GELU(), nn.Linear(d, d))
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, h, ei_j, ei_i, edge_rbf, A):
        # h (B,A,d); ei_j/ei_i (E,) index into atoms; edge_rbf (E,N_RBF)
        msg = self.filt(edge_rbf) * self.mlp(h[:, ei_j])      # (B,E,d)
        out = torch.zeros_like(h)
        out.index_add_(1, ei_i, msg)                          # scatter sum -> (B,A,d)
        return h + self.ln(out)


class AtomFlowNet(nn.Module):
    def __init__(self, d=128, n_layers=5):
        super().__init__()
        self.elem_emb = nn.Embedding(N_ELEM, d // 2)
        self.res_emb = nn.Embedding(N_TYPES, d // 2)
        self.tdim = 32
        self.inp = nn.Linear(3 + 3 + d + self.tdim, d)        # x_t + static + elem/res + time
        self.blocks = nn.ModuleList([InteractionBlock(d) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    @staticmethod
    def time_emb(t, dim=32):
        half = dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def features(self, x_t, static, elem, residx_of_atom, restype, ei_j, ei_i, edge_rbf, t):
        """Shared per-atom features (B,A,d) after interaction blocks + LayerNorm.
        Split out so the force/energy head (flow_model_ff.py) can reuse the backbone."""
        B, A, _ = x_t.shape
        emb = torch.cat([self.elem_emb(elem), self.res_emb(restype[residx_of_atom])], -1)
        te = self.time_emb(t)[:, None].expand(B, A, self.tdim)
        h = self.inp(torch.cat([x_t, static, emb, te], -1))
        for blk in self.blocks:
            h = blk(h, ei_j, ei_i, edge_rbf, A)
        return self.ln(h)

    def forward(self, x_t, static, elem, residx_of_atom, restype, ei_j, ei_i, edge_rbf, t):
        return self.out(self.features(x_t, static, elem, residx_of_atom, restype,
                                      ei_j, ei_i, edge_rbf, t))   # (B,A,3) velocity


# ---------------- data ----------------
# NOTE: 05_postprocess.py already stores atom_elements as class indices
# (0-5: C,N,O,P,S,other). Use them directly — re-mapping as atomic numbers
# silently sends every atom to "other" (5) and destroys element information.


def build_graph(static, cutoff=CUTOFF):
    """Neighbor list on static (A,3) numpy -> ei_i, ei_j (E,), edge_dist (E,)."""
    d = np.linalg.norm(static[:, None] - static[None], axis=-1)
    i, j = np.where((d < cutoff) & (d > 1e-3))
    return (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long),
            torch.tensor(d[i, j], dtype=torch.float32))


def _load_npz_system(pid, npz_path, source="md"):
    """Build a system dict from an md or static npz file."""
    z = np.load(npz_path, allow_pickle=True)
    coords, static = z["coords"].astype(np.float32), z["static"].astype(np.float32)
    ei_i, ei_j, edge_dist = build_graph(static)
    edge_rbf = dist_rbf(edge_dist)
    meta = dict(
        delta=(coords - static[None]) / SCALE,
        static=(static / SCALE).astype(np.float32),
        elem=z["atom_elements"].astype(np.int64),
        residx=z["atom_residx"].astype(np.int64),
        restype=z["res_restype"].astype(np.int64),
        edge_rbf=edge_rbf, ei_i=ei_i, ei_j=ei_j,
        pdb_id=str(pid), F=coords.shape[0], A=coords.shape[1],
        source=source,
    )
    if "atom_names" in z:
        meta["atom_names"] = z["atom_names"]
    if "source_pdb" in z:
        meta["source_pdb"] = str(z["source_pdb"])
    return meta


def load_systems(pids, static=False):
    """Per-system dict with displacement frames + a fixed static graph.

    Looks for data/md/<ID>.npz first unless static=True, then data/static/<ID>.npz.
    Skips IDs with no npz."""
    out = []
    static_dir = os.path.join(DATA, "static")
    for pid in pids:
        md_p = os.path.join(DATA, "md", f"{pid}.npz")
        st_p = os.path.join(static_dir, f"{pid}.npz")
        if static or not os.path.exists(md_p):
            if os.path.exists(st_p):
                out.append(_load_npz_system(pid, st_p, source="static"))
            elif not static and os.path.exists(md_p):
                out.append(_load_npz_system(pid, md_p, source="md"))
        else:
            out.append(_load_npz_system(pid, md_p, source="md"))
    return out


def load_split(split):
    """All selected systems in a split (thin wrapper over load_systems)."""
    import pandas as pd
    man = pd.read_csv(os.path.join(DATA, "systems_manifest.csv"))
    pids = list(man[(man.selected == 1) & (man.split == split)].pdb_id)
    return load_systems(pids)


def _to_dev(d, device):
    # static/elem/residx/restype come in as numpy from load_split; graph tensors
    # are already torch. Convert numpy -> tensor before moving to device.
    for k in ("static", "elem", "residx", "restype", "edge_rbf", "ei_i", "ei_j"):
        v = d[k]
        d[k] = (torch.from_numpy(v) if isinstance(v, np.ndarray) else v).to(device)
    return d


@torch.no_grad()
def sample_ensemble(model, sysd, n_samples=50, steps=30, sigma=1.0, device="cuda", chunk=20, chunk_auto=True):
    """Sample n_samples conformations for one system -> absolute atom coords (S,A,3) A.

    Args:
        chunk_auto: If True, automatically scale chunk size based on edge count to fit memory.
                   Target: ~400k edges per batch (conservative for 8GB GPU).
    """
    model.eval()
    if chunk_auto:
        # 目标：每批边数约 400k（保守值适配 8GB 显存）
        E = len(sysd["ei_i"])
        target_edges = 400_000
        auto_chunk = max(1, int(target_edges // E)) if E > 0 else chunk
        chunk = min(chunk, auto_chunk)
        print(f"[chunk_auto] E={E:,}, auto_chunk={auto_chunk}, using chunk={chunk}")

    outs = []
    for s in range(0, n_samples, chunk):
        b = min(chunk, n_samples - s)
        x = torch.randn(b, sysd["A"], 3, device=device) * sigma
        for k in range(steps):
            t = torch.full((b,), k / steps, device=device)
            v = model(x, sysd["static"].expand(b, -1, -1), sysd["elem"].expand(b, -1),
                      sysd["residx"].expand(b, -1), sysd["restype"], sysd["ei_j"],
                      sysd["ei_i"], sysd["edge_rbf"], t)
            x = x + v / steps
        outs.append((sysd["static"].expand(b, -1, -1) + x).cpu().numpy() * SCALE)
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--frames-per-step", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", default="",
                    help="comma-separated pdb_ids to train on (else split=train)")
    ap.add_argument("--resume", default="", help="checkpoint path to resume from")
    ap.add_argument("--save", default="",
                    help="output checkpoint path (else results/flow_model.pt)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.systems:
        pids = [s.strip().upper() for s in args.systems.split(",") if s.strip()]
        train = load_systems(pids)
    else:
        train = load_split("train")
    assert train, "no train npz — run 05_postprocess.py first"
    train = [_to_dev(t, device) for t in train]

    model = AtomFlowNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_ep = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        sigma = ckpt["sigma"]
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_ep = ckpt.get("epoch", 0)
        print(f"resumed from {args.resume} (epoch {start_ep}, sigma={sigma:.3f})")
    else:
        alld = np.concatenate([t["delta"].reshape(-1, 3) for t in train])
        sigma = float(alld.std())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"train {[t['pdb_id'] for t in train]} | frames={sum(t['F'] for t in train)} "
          f"| atoms/system={[t['A'] for t in train]} | sigma={sigma:.3f}")
    K = args.frames_per_step
    for ep in tqdm(range(args.epochs), desc="[A]train", unit="ep"):
        model.train(); tot = 0.0; nb = 0
        # shuffled full-coverage epoch: every frame seen exactly once
        all_batches = []
        for si, sysd in enumerate(train):
            idx = np.random.permutation(sysd["F"])
            for i in range(0, sysd["F"], K):
                all_batches.append((si, idx[i:i + K]))
        np.random.shuffle(all_batches)
        for si, fidx in all_batches:
            sysd = train[si]
            x1 = torch.from_numpy(sysd["delta"][fidx]).to(device)        # (K,A,3)
            B, A, _ = x1.shape
            z = torch.randn_like(x1) * sigma
            t = torch.rand(B, device=device); tm = t[:, None, None]
            x_t = (1 - tm) * z + tm * x1
            v = model(x_t, sysd["static"].expand(B, -1, -1), sysd["elem"].expand(B, -1),
                      sysd["residx"].expand(B, -1), sysd["restype"], sysd["ei_j"],
                      sysd["ei_i"], sysd["edge_rbf"], t)
            loss = ((v - (x1 - z)) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"epoch {start_ep+ep+1:4d}  loss {tot / max(nb, 1):.5f}")
    save_path = args.save if args.save else os.path.join(RESULTS, "flow_model.pt")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), sigma=sigma, config="atom-gnn-cfm",
                    opt=opt.state_dict(), epoch=start_ep + args.epochs,
                    systems=[t["pdb_id"] for t in train]),
               save_path)
    print(f"saved {save_path} (epoch {start_ep+args.epochs}, sigma={sigma:.3f}, "
          f"systems={[t['pdb_id'] for t in train]})")


if __name__ == "__main__":
    main()
