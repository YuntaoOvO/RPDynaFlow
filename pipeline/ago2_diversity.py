#!/usr/bin/env python3
"""ago2_diversity.py — hAgo2 构象集合的多样性分析（任务2 Step 7）。

对 4W5N（apo）/ 9K6T（holo）采样集合计算：
  1. pairwise RMSD 分布（集合内两两 Cα Kabsch 对齐 RMSD）
  2. 回转半径 Rg 分布（蛋白全原子 / RNA 全原子分开）
  3. PCA 前 10 PC 累计方差 + PC1-PC2 散点（Cα 构象空间）

输出：
  results/ago2/state_analysis/diversity_{png,csv}

Run:
  python3 ago2_diversity.py --samples-dir ../results/samples/ago2/r15
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ago2_state_analysis import extract_ca_coords_npz, _kabsch
from featurize_static_pdb import parse_static_pdb
from importlib import import_module

RESTYPE = import_module("05_postprocess").RESTYPE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))


def extract_rna_coords(npz_path):
    """RNA 重原子坐标 (F, Nrna, 3)。"""
    z = np.load(npz_path, allow_pickle=True)
    gen = z["gen"]
    src = str(z["source_pdb"])
    _, _, _, _, _, atoms = parse_static_pdb(src)
    rna_idx = [i for i, a in enumerate(atoms)
               if 20 <= RESTYPE.get(a["resn"], 24) < 24]
    return gen[:, rna_idx]


def extract_protein_heavy(npz_path):
    """蛋白重原子坐标（非 Cα，含侧链）(F, Nprot, 3)。"""
    z = np.load(npz_path, allow_pickle=True)
    gen = z["gen"]
    src = str(z["source_pdb"])
    _, _, _, _, _, atoms = parse_static_pdb(src)
    prot_idx = [i for i, a in enumerate(atoms)
                if RESTYPE.get(a["resn"], 24) < 20]
    return gen[:, prot_idx]


def pairwise_rmsd(ca, max_pairs=2000):
    """集合内两两 Cα Kabsch RMSD 分布（抽样 max_pairs 对控制计算量）。"""
    F = len(ca)
    pairs = []
    if F * (F - 1) // 2 > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(F * (F - 1) // 2, size=max_pairs, replace=False)
        all_pairs = [(i, j) for i in range(F) for j in range(i + 1, F)]
        chosen = [all_pairs[k] for k in idx]
    else:
        chosen = [(i, j) for i in range(F) for j in range(i + 1, F)]
    for i, j in chosen:
        aligned = _kabsch(ca[i], ca[j])
        r = np.sqrt(((aligned - ca[j]) ** 2).sum(-1).mean())
        pairs.append(r)
    return np.array(pairs)


def radius_of_gyration(coords):
    center = coords.mean(axis=1, keepdims=True)
    return np.sqrt(((coords - center) ** 2).sum(axis=-1).mean(axis=-1))


def pca_cumvar(coords, k=10):
    """Cα 构象空间 PCA：前 k PC 累计方差比例 + PC1/PC2 scores。"""
    X = coords.reshape(len(coords), -1)
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(len(X) - 1, 1)
    cum = var.cumsum() / var.sum()
    return cum[:k], U[:, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS, "ago2", "state_analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col, pid in enumerate(["4W5N", "9K6T"]):
        npz = os.path.join(args.samples_dir, f"{pid}.npz")
        ca, _ = extract_ca_coords_npz(npz)
        prot = extract_protein_heavy(npz)
        rna = extract_rna_coords(npz)
        print(f"[{pid}] Cα={ca.shape[1]} 蛋白重原子={prot.shape[1]} RNA重原子={rna.shape[1]}")

        # pairwise RMSD
        prmsd = pairwise_rmsd(ca)
        # Rg
        rg_prot = radius_of_gyration(prot)
        rg_rna = radius_of_gyration(rna)
        # PCA
        cum, scores = pca_cumvar(ca)

        axes[0, col].hist(prmsd, bins=30, color="steelblue", edgecolor="black")
        axes[0, col].axvline(prmsd.mean(), color="red", linestyle="--",
                             label=f"mean={prmsd.mean():.2f}Å")
        axes[0, col].set_title(f"{pid} pairwise Cα RMSD")
        axes[0, col].set_xlabel("RMSD (Å)")
        axes[0, col].legend()

        axes[1, col].hist(rg_prot, bins=30, alpha=0.6, label="蛋白", color="steelblue")
        axes[1, col].hist(rg_rna, bins=30, alpha=0.6, label="RNA", color="orange")
        axes[1, col].set_title(f"{pid} 回转半径 Rg")
        axes[1, col].set_xlabel("Rg (Å)")
        axes[1, col].legend()

        rows.append(dict(
            pdb_id=pid,
            n_frames=int(ca.shape[0]),
            pairwise_rmsd_mean=float(prmsd.mean()),
            pairwise_rmsd_std=float(prmsd.std()),
            rg_protein_mean=float(rg_prot.mean()),
            rg_protein_std=float(rg_prot.std()),
            rg_rna_mean=float(rg_rna.mean()),
            rg_rna_std=float(rg_rna.std()),
            pca_pc1_var=float(cum[0]),
            pca_pc2_var=float(cum[1]),
            pca_top5_var=float(cum[4]),
            pca_top10_var=float(cum[9]),
        ))

    # PCA 散点（公共 Cα 对齐后联合投影）
    ax = axes[0, 2] if False else axes[0, 2]
    # 用各自的 PC1-PC2 简单叠加（标注不同系统）
    colors = {"4W5N": "steelblue", "9K6T": "orange"}
    for pid in ["4W5N", "9K6T"]:
        npz = os.path.join(args.samples_dir, f"{pid}.npz")
        ca, _ = extract_ca_coords_npz(npz)
        _, scores = pca_cumvar(ca)
        axes[0, 2].scatter(scores[:, 0], scores[:, 1], s=8, alpha=0.5,
                           color=colors[pid], label=pid)
    axes[0, 2].set_title("PCA PC1-PC2 (各系统独立)")
    axes[0, 2].set_xlabel("PC1")
    axes[0, 2].set_ylabel("PC2")
    axes[0, 2].legend()

    fig.delaxes(axes[1, 2])
    fig.tight_layout()
    out_png = os.path.join(args.out, "diversity.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "diversity_summary.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\n写入 -> {args.out}")


if __name__ == "__main__":
    main()
