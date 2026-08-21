#!/usr/bin/env python3
"""visualize_esmdynamic_overlay.py — ESMDynamic 接触预测 + DynaFlow 系综验证

功能：
  1. 加载 ESMDynamic 输出（dynamic_prob, frequency_pred, kinetics）
  2. 从 DynaFlow 系综计算实际接触频率（8Å Cα cutoff）
  3. 可视化对比：预测 vs 实际（热图、差异图、ROC）
  4. 残基级动态评分：与 ESMDynamic confidence 对比

Run:
  python3 visualize_esmdynamic_overlay.py \
    --dynaflow ../results/samples/ago2/r15/4W5N.npz \
    --esmdynamic ../data/esmdynamic_outputs/4W5N \
    --out ../results/esmdynamic_overlay/4W5N
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esmdynamic_overlay import load_esmdynamic_frequency, load_esmdynamic_dynamic_prob
from featurize_static_pdb import parse_static_pdb, _find_pdb
from importlib import import_module

RESTYPE = import_module("05_postprocess").RESTYPE

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
CONTACT_CUTOFF = 8.0  # Å（与 ESMDynamic 一致）


def compute_contact_frequency(coords, cutoff=CONTACT_CUTOFF):
    """coords (F,N,3) -> (N,N) 接触频率矩阵"""
    F, N, _ = coords.shape
    freq = np.zeros((N, N), dtype=np.float32)
    for f in range(F):
        d = cdist(coords[f], coords[f])
        contact = (d < cutoff) & (d > 1e-3)
        freq += contact.astype(np.float32)
    freq /= F
    return freq


def load_dynaflow_ca_contacts(npz_path):
    """从 DynaFlow npz 提取 CA 接触频率"""
    z = np.load(npz_path, allow_pickle=True)
    gen = z["gen"]
    src = str(z.get("source_pdb", ""))
    if not src or not os.path.exists(src):
        pid = os.path.basename(npz_path).replace(".npz", "")
        src = _find_pdb(pid)

    _, _, _, res_rt, _, atoms = parse_static_pdb(src)
    ca_idx = [i for i, a in enumerate(atoms) if a["name"] == "CA"
              and RESTYPE.get(a["resn"], 24) < 20]
    ca_coords = gen[:, ca_idx]
    freq = compute_contact_frequency(ca_coords)
    return freq, ca_idx


def plot_heatmap_comparison(pred, obs, out_path, title="ESMDynamic vs DynaFlow"):
    """3 子图：预测、实际、差异"""
    L = min(pred.shape[0], obs.shape[0])
    pred = pred[:L, :L]
    obs = obs[:L, :L]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(pred, origin="lower", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("ESMDynamic 预测频率")
    axes[0].set_xlabel("残基")
    axes[0].set_ylabel("残基")
    plt.colorbar(im0, ax=axes[0], label="频率")

    im1 = axes[1].imshow(obs, origin="lower", cmap="Oranges", vmin=0, vmax=1)
    axes[1].set_title("DynaFlow 实际频率")
    axes[1].set_xlabel("残基")
    plt.colorbar(im1, ax=axes[1], label="频率")

    diff = pred - obs
    im2 = axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[2].set_title("差异（预测 - 实际）")
    axes[2].set_xlabel("残基")
    plt.colorbar(im2, ax=axes[2], label="Δ频率")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 热图对比: {out_path}")


def plot_roc(pred, obs, out_path, threshold=0.1):
    """ROC 曲线：二分类（接触 vs 非接触）"""
    mask = ~np.eye(pred.shape[0], dtype=bool)  # 排除对角线
    y_true = (obs[mask] > threshold).astype(int)
    y_score = pred[mask]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机分类器")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC 曲线（接触阈值 > {threshold}）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] ROC 曲线 (AUC={roc_auc:.3f}): {out_path}")
    return roc_auc


def residue_dynamics_score(contact_freq):
    """残基级动态性评分：接触伙伴数的变异系数"""
    # 每残基的接触频率分布的熵（粗略量化动态性）
    scores = np.zeros(contact_freq.shape[0])
    for i in range(contact_freq.shape[0]):
        partners = contact_freq[i]
        partners = partners[partners > 0.01]  # 过滤噪声
        if len(partners) > 0:
            scores[i] = partners.std() / (partners.mean() + 1e-6)
    return scores


def plot_residue_dynamics(dynaflow_scores, esm_confidence, out_path):
    """残基动态性评分对比"""
    L = min(len(dynaflow_scores), len(esm_confidence))
    x = np.arange(L)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(x, dynaflow_scores[:L], lw=1.5, color="#3C5A78", label="DynaFlow")
    axes[0].set_ylabel("变异系数（接触频率）")
    axes[0].set_title("残基级动态性评分")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, esm_confidence[:L], lw=1.5, color="#D97D54", label="ESMDynamic confidence")
    axes[1].set_xlabel("残基索引")
    axes[1].set_ylabel("Confidence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 残基动态性: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynaflow", required=True, help="DynaFlow npz (e.g., 4W5N.npz)")
    ap.add_argument("--esmdynamic", required=True, help="ESMDynamic output dir (e.g., 4W5N/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp", type=int, default=320, help="ESMDynamic temperature (K)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pid = os.path.basename(args.dynaflow).replace(".npz", "")
    print(f"[系统] {pid}")
    print(f"  DynaFlow: {args.dynaflow}")
    print(f"  ESMDynamic: {args.esmdynamic}")
    print(f"  输出: {args.out}\n")

    # 加载 DynaFlow 接触频率
    df_freq, ca_idx = load_dynaflow_ca_contacts(args.dynaflow)
    print(f"[DynaFlow] CA 接触矩阵: {df_freq.shape}")

    # 加载 ESMDynamic 预测
    esm_freq = load_esmdynamic_frequency(args.esmdynamic, pid, temp_k=args.temp)
    esm_prob = load_esmdynamic_dynamic_prob(args.esmdynamic, pid, temp_k=args.temp)

    if esm_freq is None:
        sys.exit(f"[ERROR] ESMDynamic frequency 文件不存在: {args.esmdynamic}")

    print(f"[ESMDynamic] 预测矩阵: {esm_freq.shape}")

    # 尺寸对齐（取最小）
    L = min(df_freq.shape[0], esm_freq.shape[0])
    df_freq = df_freq[:L, :L]
    esm_freq = esm_freq[:L, :L]

    # 热图对比
    plot_heatmap_comparison(esm_freq, df_freq,
                            os.path.join(args.out, "heatmap_comparison.png"),
                            title=f"{pid} @ {args.temp}K")

    # ROC 曲线
    roc_auc = plot_roc(esm_freq, df_freq,
                       os.path.join(args.out, "roc_curve.png"),
                       threshold=0.1)

    # 残基动态性
    df_scores = residue_dynamics_score(df_freq)
    esm_conf = esm_prob.mean(axis=1) if esm_prob is not None else np.ones(L) * 0.5
    plot_residue_dynamics(df_scores, esm_conf,
                          os.path.join(args.out, "residue_dynamics.png"))

    # 汇总指标
    mae = np.abs(esm_freq - df_freq).mean()
    corr = np.corrcoef(esm_freq.flatten(), df_freq.flatten())[0, 1]

    print(f"\n[指标]")
    print(f"  MAE（频率）: {mae:.4f}")
    print(f"  Pearson 相关: {corr:.3f}")
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"\n写入 -> {args.out}")

    # 保存指标
    with open(os.path.join(args.out, "metrics.txt"), "w") as f:
        f.write(f"system: {pid}\n")
        f.write(f"temperature: {args.temp}K\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"Pearson_corr: {corr:.3f}\n")
        f.write(f"ROC_AUC: {roc_auc:.3f}\n")


if __name__ == "__main__":
    main()
