#!/usr/bin/env python3
"""sanity_check.py — validate the atomic GNN end-to-end on one train npz.

Loads one train system, runs one forward+backward and one sample_ensemble,
and checks: finite loss, finite samples, AND that element classes are a real
mix (not all "other"/5 — the _elem_class double-encoding bug).

Run:  python3 sanity_check.py        (needs data/md/<ID>.npz for >=1 train system)
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_model import AtomFlowNet, load_split, sample_ensemble, _to_dev

np.random.seed(0); torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  cuda_avail={torch.cuda.is_available()}")

train = load_split("train")
assert train, "no train npz in data/md/ — run 05_postprocess.py --only <train> first"
sysd = _to_dev(train[0], device)
print(f"loaded {sysd['pdb_id']}: A={sysd['A']} F={sysd['F']} "
      f"edges={sysd['ei_i'].numel()}")

# --- element-class sanity (the bug we fixed) ---
uc = sysd['elem'].bincount(minlength=6).cpu().numpy()
print(f"element histogram [C,N,O,P,S,other] = {uc.tolist()}")
assert uc[:5].sum() > 0, "BUG: every atom is 'other'(5) — element info lost (double-encoding?)"

# --- one forward + backward ---
model = AtomFlowNet().to(device)
K = 8
fidx = np.random.choice(sysd['F'], size=min(K, sysd['F']), replace=False)
x1 = torch.from_numpy(sysd['delta'][fidx]).to(device)
B = x1.shape[0]
sigma = float(sysd['delta'].std())
z = torch.randn_like(x1) * sigma
t = torch.rand(B, device=device); tm = t[:, None, None]
x_t = (1 - tm) * z + tm * x1
v = model(x_t, sysd['static'].expand(B, -1, -1), sysd['elem'].expand(B, -1),
          sysd['residx'].expand(B, -1), sysd['restype'], sysd['ei_j'],
          sysd['ei_i'], sysd['edge_rbf'], t)
loss = ((v - (x1 - z)) ** 2).mean()
loss.backward()
gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
n_params = sum(p.numel() for p in model.parameters())
print(f"params={n_params/1e6:.2f}M  loss={loss.item():.5f} "
      f"finite={torch.isfinite(loss).item()}  grad_norm~={gnorm:.2f}")
assert torch.isfinite(loss), "NaN/inf in loss!"

# --- sample ---
gen = sample_ensemble(model, sysd, n_samples=4, steps=10, sigma=sigma,
                      device=device, chunk=4)
print(f"sample: shape={gen.shape} range=[{gen.min():.2f},{gen.max():.2f}] "
      f"finite={np.isfinite(gen).all()}")
assert np.isfinite(gen).all(), "NaN in samples!"
print("SMOKE TEST PASSED")
