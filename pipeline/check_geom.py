#!/usr/bin/env python3
"""check_geom.py — covalent-geometry sanity of a generated ensemble vs its MD.

Bonds are inferred from the static structure (heavy-atom pairs < 1.8 A), which is
topology-free and therefore works for any system. MD is the control: MD's own bond
deviation is the noise floor any generator must reach.

Run: python3 check_geom.py <samples_dir> [pdb_ids...]
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ROOT", os.path.dirname(HERE))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))


def bonds_from_static(st, cutoff=1.8):
    d = np.linalg.norm(st[:, None] - st[None], axis=-1)
    iu = np.triu_indices(len(st), k=1)
    m = d[iu] < cutoff
    return iu[0][m], iu[1][m], d[iu][m]


def bond_stats(C, i, j, ref):
    gd = np.linalg.norm(C[:, i] - C[:, j], axis=-1)
    dev = np.abs(gd - ref[None])
    return dict(mean_dev=float(dev.mean()), frac_bad=float((dev > 0.3).mean()),
                max_dev=float(dev.max()))


def rmsf(c):
    return np.sqrt(((c - c.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))


def main():
    sdir = sys.argv[1]
    pids = sys.argv[2:] or sorted(
        f[:-4] for f in os.listdir(sdir) if f.endswith(".npz"))
    print(f"{'system':8s} {'in_tr':5s} | {'MD dev':>7s} {'GEN dev':>7s} "
          f"{'GEN bad':>7s} {'GEN max':>7s} | {'rmsf_md':>7s} {'rmsf_gen':>8s} {'r':>6s}")
    print("-" * 88)
    for pid in pids:
        sp = os.path.join(sdir, f"{pid}.npz")
        mp = os.path.join(DATA, "md", f"{pid}.npz")
        if not (os.path.exists(sp) and os.path.exists(mp)):
            print(f"{pid:8s} missing npz"); continue
        s, z = np.load(sp, allow_pickle=True), np.load(mp)
        gen = s["gen"].astype(np.float64)
        md = z["coords"].astype(np.float64)
        st = z["static"].astype(np.float64)
        i, j, ref = bonds_from_static(st)
        bm, bg = bond_stats(md, i, j, ref), bond_stats(gen, i, j, ref)
        rm, rg = rmsf(md), rmsf(gen)
        a, b = rm - rm.mean(), rg - rg.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        r = float((a * b).sum() / den) if den > 0 else 0.0
        intr = bool(s["in_train"]) if "in_train" in s else False
        print(f"{pid:8s} {str(intr):5s} | {bm['mean_dev']:7.3f} {bg['mean_dev']:7.3f} "
              f"{bg['frac_bad']:7.3f} {bg['max_dev']:7.2f} | {rm.mean():7.2f} "
              f"{rg.mean():8.2f} {r:6.3f}")
    print("\nMD dev = noise floor (thermal bond vibration, expect ~0.03 A).")
    print("GEN bad = fraction of bonds off by >0.3 A. GEN max = worst single bond.")


if __name__ == "__main__":
    main()
