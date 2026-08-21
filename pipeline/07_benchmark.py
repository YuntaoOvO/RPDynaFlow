#!/usr/bin/env python3
"""07_benchmark.py — evaluate the atomic flow-matching ensemble generator.

Per TEST system: sample n_gen conformations, compare to held-out MD (and NMR when
available) via:
  - RMSF Pearson + mean amplitude          (flexibility pattern)
  - RMSD-to-static Wasserstein             (exploration range)
  - PCA coverage Recall/Precision/F1 + JS  (conformational-space overlap)
  - force Pearson/RMSE (B/C only)          (learned force-field quality, where md_ff exists)
  - PCA scatter figure

Model A (AtomFlowNet) or B (AtomFlowNetFF) auto-detected from ckpt["config"].
--guided ETA switches to energy-guided sampling (Model C).

Run:  python3 07_benchmark.py --ckpt results/checkpoints/flow_model_r3.pt --n-gen 50
      python3 07_benchmark.py --ckpt results/ff/flow_model_ff.pt --guided 0.1
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from flow_model import (AtomFlowNet, sample_ensemble, load_split, load_systems,
                        _to_dev, DATA, RESULTS, SCALE)


# ---------------- metrics ----------------
def rmsf(coords):
    mu = coords.mean(0, keepdims=True)
    return np.sqrt(((coords - mu) ** 2).sum(-1).mean(0))

def rmsd_to(coords, ref):
    return np.sqrt(((coords - ref[None]) ** 2).sum(-1).mean(-1))

def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def wasserstein_1d(a, b, n=200):
    qs = np.linspace(0, 1, n)
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())

def pca_project(md, gen, k=2):
    X = md.reshape(len(md), -1); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:k].T
    return (X - mu) @ P, (gen.reshape(len(gen), -1) - mu) @ P

def compute_coverage(md_pc, gen_pc, bins=60, min_counts=3):
    """Recall/Precision/F1: does gen cover MD's PC1-PC2 bins (MD defines the space)?"""
    xe = np.linspace(md_pc[:, 0].min(), md_pc[:, 0].max(), bins + 1)
    ye = np.linspace(md_pc[:, 1].min(), md_pc[:, 1].max(), bins + 1)
    Hm, _, _ = np.histogram2d(md_pc[:, 0], md_pc[:, 1], bins=[xe, ye])
    Hg, _, _ = np.histogram2d(gen_pc[:, 0], gen_pc[:, 1], bins=[xe, ye])
    Mm, Mg = Hm >= min_counts, Hg >= 1
    inter = int((Mm & Mg).sum())
    rec = inter / int(Mm.sum()) if Mm.sum() else 0.0
    prec = inter / int(Mg.sum()) if Mg.sum() else 0.0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
    return float(rec), float(prec), float(f1)

def js_2d(md_pc, gen_pc, bins=50):
    """Jensen-Shannon divergence of the PC1-PC2 distributions (parameter-free)."""
    lo = (min(md_pc[:, 0].min(), gen_pc[:, 0].min()), min(md_pc[:, 1].min(), gen_pc[:, 1].min()))
    hi = (max(md_pc[:, 0].max(), gen_pc[:, 0].max()), max(md_pc[:, 1].max(), gen_pc[:, 1].max()))
    xe = np.linspace(lo[0], hi[0], bins + 1); ye = np.linspace(lo[1], hi[1], bins + 1)
    p, _, _ = np.histogram2d(md_pc[:, 0], md_pc[:, 1], bins=[xe, ye])
    q, _, _ = np.histogram2d(gen_pc[:, 0], gen_pc[:, 1], bins=[xe, ye])
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12); m = 0.5 * (p + q)
    def kl(a, b): return float((a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

@torch.no_grad()
def force_diagnostic(model, sysd, device, k_frames=64):
    """Force-matching quality (B/C only): F_pred = -grad(E)/SCALE vs stored MD forces.
    Returns (pearson, rmse); (None, None) if no energy head or no md_ff data for system.
    NB: energy is evaluated at t=0 (the head is weakly t-conditioned via features(); the
    force-matching target is t-independent, so this is a representative snapshot)."""
    if not hasattr(model, "energy"):
        return None, None
    ff_p = os.path.join(DATA, "md_ff", f"{sysd['pdb_id']}.npz")
    if not os.path.exists(ff_p):
        return None, None
    forces = np.load(ff_p)["forces"].astype(np.float32)        # (F,A,3) physical, aligned
    delta = sysd["delta"]                                        # (F,A,3) numpy (md==md_ff coords)
    fi = np.random.choice(delta.shape[0], size=min(k_frames, delta.shape[0]), replace=False)
    # NOTE: temporarily enable grad for the energy pass
    with torch.enable_grad():
        x1 = torch.from_numpy(delta[fi]).to(device).requires_grad_(True)
        B = x1.shape[0]; t = torch.zeros(B, device=device)
        E = model.energy(x1, sysd["static"].expand(B, -1, -1), sysd["elem"].expand(B, -1),
                         sysd["residx"].expand(B, -1), sysd["restype"], sysd["ei_j"],
                         sysd["ei_i"], sysd["edge_rbf"], t)
        g = torch.autograd.grad(E.sum(), x1)[0]
    F_pred = (-g / SCALE).detach().cpu().numpy()
    F_md = forces[fi]
    return pearson(F_pred.flatten(), F_md.flatten()), float(np.sqrt(((F_pred - F_md) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=50)
    ap.add_argument("--ckpt", default="", help="checkpoint path (else results/flow_model.pt)")
    ap.add_argument("--systems", default="", help="comma-sep pdb_ids (else split=test)")
    ap.add_argument("--out", default="", help="output CSV (else results/benchmark_summary.csv)")
    ap.add_argument("--guided", type=float, default=0.0,
                    help="eta for energy-guided sampling (Model C); 0 = plain sampling")
    ap.add_argument("--dump-samples", action="store_true",
                    help="save generated coords to results/samples/<ckpt>/<PDB>.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt if args.ckpt else os.path.join(RESULTS, "flow_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", "")
    if "ff" in cfg:                       # Model B/C checkpoint (has energy head)
        from flow_model_ff import AtomFlowNetFF
        model = AtomFlowNetFF().to(device)
    else:                                 # Model A
        model = AtomFlowNet().to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    sigma = ckpt["sigma"]
    print(f"ckpt {ckpt_path} | config={cfg or '?'} | sigma={sigma:.3f} | "
          f"trained_on={ckpt.get('systems', [])} | guided_eta={args.guided}")

    if args.systems:
        pids = [s.strip().upper() for s in args.systems.split(",") if s.strip()]
        test = load_systems(pids)
    else:
        test = load_split("test")
    test = [_to_dev(s, device) for s in test]
    figdir = os.path.join(RESULTS, "figs"); os.makedirs(figdir, exist_ok=True)
    if args.dump_samples:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        if args.guided > 0:
            ckpt_stem += f"_guided{args.guided}"
        sampdir = os.path.join(RESULTS, "samples", ckpt_stem)
        os.makedirs(sampdir, exist_ok=True)
    rows = []
    for sysd in test:
        pid = sysd["pdb_id"]
        md = (sysd["delta"] + sysd["static"].cpu().numpy()) * SCALE
        static_abs = sysd["static"].cpu().numpy() * SCALE
        try:
            if args.guided > 0:
                from flow_model_ff import sample_ensemble_guided
                gen = sample_ensemble_guided(model, sysd, n_samples=args.n_gen, sigma=sigma,
                                             eta=args.guided, device=device)
            else:
                gen = sample_ensemble(model, sysd, n_samples=args.n_gen, sigma=sigma, device=device)
        except Exception as e:
            print(f"[sample-fail] {pid}: {e}"); continue
        if args.dump_samples:
            np.savez_compressed(os.path.join(sampdir, f"{pid}.npz"),
                               gen_coords=gen, static=static_abs, pdb_id=pid)
        r_md, r_gen = rmsf(md), rmsf(gen)
        d_md, d_gen = rmsd_to(md, static_abs), rmsd_to(gen, static_abs)
        W, G = pca_project(md, gen)
        rec, prec, f1 = compute_coverage(W, G)
        js = js_2d(W, G)
        fp, fr = force_diagnostic(model, sysd, device)
        row = dict(pdb_id=pid, n_atoms=int(md.shape[1]),
                   rmsf_pearson=round(pearson(r_md, r_gen), 3),
                   mean_rmsf_md=round(float(r_md.mean()), 3),
                   mean_rmsf_gen=round(float(r_gen.mean()), 3),
                   rmsd_wass=round(wasserstein_1d(d_md, d_gen), 2),
                   cov_recall=round(rec, 3), cov_precision=round(prec, 3), cov_f1=round(f1, 3),
                   js_pc=round(js, 3))
        if fp is not None:
            row.update(force_pearson=round(fp, 3), force_rmse=round(fr, 1))
        # NMR ground truth (atom sets can differ from MD: NMR parsing skips HETATM
        # and stale npz may be residue-level — only compare when atom counts match)
        nmr_p = os.path.join(DATA, "nmr", f"{pid}.npz")
        r_nmr = None
        if os.path.exists(nmr_p):
            r_nmr = rmsf(np.load(nmr_p)["coords"])
            if len(r_nmr) != len(r_gen):
                print(f"[nmr-skip] {pid}: nmr atoms {len(r_nmr)} != md/gen {len(r_gen)}")
                r_nmr = None
            else:
                row.update(rmsf_pearson_nmr=round(pearson(r_nmr, r_gen), 3))
        rows.append(row); print(pid, row)

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        x = np.arange(len(r_md))
        ax[0].plot(x, r_md, lw=1.4, label="MD")
        ax[0].plot(x, r_gen, lw=1.4, alpha=.85, label="Flow")
        if r_nmr is not None:
            ax[0].plot(x, r_nmr, lw=1.2, ls="--", label="NMR")
        ax[0].set_xlabel("atom index"); ax[0].set_ylabel("RMSF (A)"); ax[0].legend(fontsize=8)
        ax[0].set_title(f"{pid} atom RMSF")
        ax[1].scatter(W[:, 0], W[:, 1], s=5, alpha=.4, label="MD")
        ax[1].scatter(G[:, 0], G[:, 1], s=5, alpha=.4, label="Flow")
        ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{pid}.png"), dpi=150); plt.close(fig)

    if not rows:
        print("nothing to benchmark"); return
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = args.out if args.out else os.path.join(RESULTS, "benchmark_summary.csv")
    df.to_csv(out_csv, index=False)
    print("\n===== summary =====\n" + df.to_string(index=False))
    print(f"\nmean atom-RMSF pearson vs MD: {df.rmsf_pearson.mean():.3f} | "
          f"mean cov_F1: {df.cov_f1.mean():.3f} | mean JS: {df.js_pc.mean():.3f}")
    if "rmsf_pearson_nmr" in df:
        sub = df.dropna(subset=["rmsf_pearson_nmr"])
        if len(sub):
            print(f"mean atom-RMSF pearson vs NMR: {sub.rmsf_pearson_nmr.mean():.3f}")
    if "force_pearson" in df:
        sub = df.dropna(subset=["force_pearson"])
        if len(sub):
            print(f"mean force pearson (B/C): {sub.force_pearson.mean():.3f} | "
                  f"rmse: {sub.force_rmse.mean():.1f} kJ/mol/Å")


if __name__ == "__main__":
    main()
