#!/usr/bin/env python3
"""bioemu_bond_sanity.py — 键长保持的量化分析（核心物理有效性发现）。

对比训练数据（MD 轨迹，键长由力场约束）与模型采样产物的键长断裂率，
覆盖 hAgo2 复合物（r5/r10/r15）与 benchmark 单体（采样完成后）。

键长判据：静态结构中距离 < 1.8 Å 的重原子对视为共价键，采样后距离
  - 断裂：> 2.5 Å
  - 坍缩：< 1.0 Å
官方 unphysical 过滤的 np.all 语义下，任一断裂键即排除该样本，
因此"含 >=1 断裂键的样本比例"直接决定官方过滤的通过率。

Run:
  python3 bioemu_bond_sanity.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def bond_breakage(coords, static, cutoff=1.8, break_thr=2.5, collapse_thr=1.0):
    """coords (F,A,3), static (A,3) -> 键长断裂统计 dict."""
    d = np.linalg.norm(static[:, None] - static[None], axis=-1)
    iu = np.triu_indices(len(static), k=1)
    m = d[iu] < cutoff
    i, j = iu[0][m], iu[1][m]
    if len(i) == 0:
        return dict(n_bonds=0)
    ref = d[i, j]
    gd = np.linalg.norm(coords[:, i] - coords[:, j], axis=-1)   # (F, nb)
    return dict(
        n_bonds=int(len(i)),
        ref_len=float(ref.mean()),
        gen_len=float(gd.mean()),
        break_rate=float((gd > break_thr).mean()),
        collapse_rate=float((gd < collapse_thr).mean()),
        any_broken_rate=float((gd > break_thr).any(axis=1).mean()),
    )


def analyze_npz(npz, label):
    z = np.load(npz, allow_pickle=True)
    coords = z["coords"] if "coords" in z else z["gen"]
    static = z["static"]
    r = bond_breakage(coords, static)
    r["label"] = label
    r["n_frames"] = int(coords.shape[0])
    r["n_atoms"] = int(static.shape[0])
    return r


def main():
    rows = []
    # 1. 训练数据 baseline（MD 轨迹，键长由力场约束）
    for pid in ["1NYB", "1EKZ", "2ESE"]:
        p = os.path.join(DATA, "md", f"{pid}.npz")
        if os.path.exists(p):
            rows.append(analyze_npz(p, f"训练数据 {pid}"))

    # 2. hAgo2 复合物（r5/r10/r15 × 4W5N/9K6T）
    for ckpt in ["r5", "r10", "r15"]:
        for pid in ["4W5N", "9K6T"]:
            p = os.path.join(RESULTS, "samples", "ago2", ckpt, f"{pid}.npz")
            if os.path.exists(p):
                rows.append(analyze_npz(p, f"hAgo2 {pid} {ckpt}"))

    # 3. benchmark 单体（采样完成后 samples_raw/*/<TC>.npz）
    raw = os.path.join(RESULTS, "ago2", "bioemu_bench", "samples_raw")
    for npz in sorted(glob.glob(os.path.join(raw, "*", "*.npz"))):
        pid = os.path.basename(npz)[:-4]
        rows.append(analyze_npz(npz, f"bench {pid}"))

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "ago2", "bond_sanity.csv")
    df.to_csv(out, index=False)
    cols = ["label", "n_frames", "n_atoms", "n_bonds", "ref_len", "gen_len",
            "break_rate", "collapse_rate", "any_broken_rate"]
    print(df[cols].to_string(index=False))
    print(f"\n写入 -> {out}")


if __name__ == "__main__":
    main()
