#!/usr/bin/env python3
"""ago2_state_analysis.py — 状态特异性分析（RMSD 分布、接触差异、结构域拆分）

扩展 ago2_multiconf_eval.py，增加：
  A. RMSD 分布直方图（4 子图：自 RMSD vs 跨结构 RMSD）
  B. 结构域拆分 RMSD 条形图
  C. 接触图差异（RNA-蛋白界面，9K6T - 4W5N）
  D. 零假设对比（高斯噪声采样）

Run:
  python3 ago2_state_analysis.py --samples-dir ../results/samples/ago2/r15
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize_static_pdb import parse_static_pdb, _find_pdb
from importlib import import_module


def _setup_cjk_font():
    """Best-effort CJK font setup for Chinese plot labels (no-op if unavailable)."""
    import matplotlib.font_manager as fm
    _font_files = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    _font_names = ["Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei",
                   "AR PL UMing CN", "SimHei", "SimSun"]
    try:
        for fp in _font_files:
            if os.path.exists(fp):
                fm.fontManager.addfont(fp)
        installed = {f.name for f in fm.fontManager.ttflist}
        for name in _font_names:
            if name in installed:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return
    except Exception:
        pass


_setup_cjk_font()

RESTYPE = import_module("05_postprocess").RESTYPE

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))

AGO2_DOMAINS = {
    "N": (1, 61),
    "L1": (62, 182),
    "PAZ": (183, 277),
    "L2": (278, 421),
    "MID": (422, 575),
    "PIWI": (576, 859),
}

CONTACT_CUTOFF = 8.0  # Å


def _kabsch(mobile, ref):
    mc, rc = mobile.mean(0), ref.mean(0)
    H = (mobile - mc).T @ (ref - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (mobile - mc) @ R.T + rc


def extract_ca_coords_npz(npz_path):
    """Extract protein CA coordinates (F,N,3) from gen npz."""
    z = np.load(npz_path, allow_pickle=True)
    gen = z["gen"]
    src = str(z["source_pdb"]) if "source_pdb" in z else ""
    if not src or not os.path.exists(src):
        pid = os.path.basename(npz_path).replace(".npz", "")
        src = _find_pdb(pid)
    _, _, _, res_rt, _, atoms = parse_static_pdb(src)
    ca_idx = [i for i, a in enumerate(atoms) if a["name"] == "CA"
              and RESTYPE.get(a["resn"], 24) < 20]
    ca_coords = gen[:, ca_idx]
    resnums = [int(atoms[i]["resseq"]) for i in ca_idx]
    return ca_coords, resnums


def rmsd_to_ref(gen_ca, ref_ca):
    """gen_ca (F,N,3), ref_ca (N,3) -> (F,) RMSD after per-frame Kabsch."""
    out = np.empty(len(gen_ca), dtype=np.float64)
    for i in range(len(gen_ca)):
        aligned = _kabsch(gen_ca[i], ref_ca)
        out[i] = np.sqrt(((aligned - ref_ca) ** 2).sum(-1).mean())
    return out


def plot_rmsd_distributions(rmsds_dict, out_path):
    """4 子图：对角线（自 RMSD）vs 交叉项（跨结构 RMSD）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    bins = np.arange(0, 15.5, 0.5)

    titles = [
        ("4W5N_gen→4W5N_ref (自 RMSD)", "4W5N_self"),
        ("4W5N_gen→9K6T_ref (跨结构)", "4W5N_to_9K6T"),
        ("9K6T_gen→4W5N_ref (跨结构)", "9K6T_to_4W5N"),
        ("9K6T_gen→9K6T_ref (自 RMSD)", "9K6T_self"),
    ]

    for ax, (title, key) in zip(axes.flat, titles):
        if key not in rmsds_dict:
            ax.text(0.5, 0.5, "数据缺失", ha="center", va="center")
            ax.set_title(title)
            continue
        data = rmsds_dict[key]
        ax.hist(data, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(data.mean(), color="red", linestyle="--", label=f"mean={data.mean():.2f}Å")
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("频数")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] RMSD 分布直方图: {out_path}")


def domain_rmsd_analysis(gen_ca, ref_ca, resnums):
    """结构域拆分 RMSD"""
    results = {}
    for dname, (lo, hi) in AGO2_DOMAINS.items():
        mask = np.array([lo <= r <= hi for r in resnums], dtype=bool)
        if mask.sum() < 5:
            continue
        sub_gen = gen_ca[:, mask]
        sub_ref = ref_ca[mask]
        rmsds = rmsd_to_ref(sub_gen, sub_ref)
        results[dname] = float(rmsds.mean())
    return results


def plot_domain_rmsd(domains_dict, out_path):
    """结构域条形图：4W5N/9K6T 对两个参考结构的响应"""
    df = pd.DataFrame(domains_dict).T
    df.plot(kind="bar", figsize=(10, 6), rot=0, width=0.8)
    plt.xlabel("结构域")
    plt.ylabel("Mean RMSD (Å)")
    plt.title("AGO2 结构域 RMSD（apo vs holo 参考）")
    plt.legend(title="系综 → 参考")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[图] 结构域 RMSD 条形图: {out_path}")


def contact_frequency(coords, cutoff=CONTACT_CUTOFF):
    """coords (F,N,3) -> (N,N) 接触频率矩阵"""
    F, N, _ = coords.shape
    freq = np.zeros((N, N), dtype=np.float32)
    for f in range(F):
        d = cdist(coords[f], coords[f])
        contact = (d < cutoff) & (d > 1e-3)
        freq += contact.astype(np.float32)
    freq /= F
    freq = freq + freq.T  # 对称化
    np.fill_diagonal(freq, 0)
    return freq


def plot_contact_diff(freq_a, freq_b, out_path, label_a="4W5N", label_b="9K6T"):
    """接触频率差异热图：|freq_b - freq_a|"""
    L = min(freq_a.shape[0], freq_b.shape[0])
    diff = np.abs(freq_b[:L, :L] - freq_a[:L, :L])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(diff, origin="lower", cmap="Reds", vmin=0, vmax=0.5)
    ax.set_title(f"接触频率差异 |{label_b} - {label_a}|")
    ax.set_xlabel("残基索引")
    ax.set_ylabel("残基索引")
    plt.colorbar(im, ax=ax, label="Δ频率")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 接触差异热图: {out_path}")


def null_baseline(static_coords, n_samples=200, noise_std=2.0):
    """零假设：高斯噪声扰动（单位：Å）"""
    N = static_coords.shape[0]
    noise = np.random.randn(n_samples, N, 3) * noise_std
    return static_coords[None] + noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--out", default="", help="default: results/ago2/<basename>_state_analysis")
    ap.add_argument("--pdb-a", default="4W5N")
    ap.add_argument("--pdb-b", default="9K6T")
    args = ap.parse_args()

    pdb_a = _find_pdb(args.pdb_a)
    pdb_b = _find_pdb(args.pdb_b)
    if not pdb_a or not pdb_b:
        sys.exit(f"missing PDB: {args.pdb_a}={pdb_a} {args.pdb_b}={pdb_b}")

    npz_a = os.path.join(args.samples_dir, f"{args.pdb_a}.npz")
    npz_b = os.path.join(args.samples_dir, f"{args.pdb_b}.npz")
    for p in (npz_a, npz_b):
        if not os.path.exists(p):
            sys.exit(f"missing sample npz: {p}")

    out_dir = args.out or os.path.join(
        RESULTS, "ago2", os.path.basename(args.samples_dir.rstrip("/")) + "_state_analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[分析] 样本目录: {args.samples_dir}")
    print(f"[分析] 输出: {out_dir}\n")

    # 提取 CA 坐标
    gen_a, resnums_a = extract_ca_coords_npz(npz_a)
    gen_b, resnums_b = extract_ca_coords_npz(npz_b)

    # 参考结构（PDB frame 0）
    from ago2_multiconf_eval import extract_ca_map, common_residues, ca_vectors
    ca_map_a = extract_ca_map(pdb_a)
    ca_map_b = extract_ca_map(pdb_b)
    common_res = common_residues(ca_map_a, ca_map_b)
    ref_a = ca_vectors(ca_map_a, common_res)
    ref_b = ca_vectors(ca_map_b, common_res)

    # 对齐残基索引（取公共部分）
    idx_a = [i for i, r in enumerate(resnums_a) if r in common_res]
    idx_b = [i for i, r in enumerate(resnums_b) if r in common_res]
    gen_a_aligned = gen_a[:, idx_a]
    gen_b_aligned = gen_b[:, idx_b]

    print(f"[对齐] 公共 CA 残基: {len(common_res)} ({common_res[0]}..{common_res[-1]})")

    # A. RMSD 分布
    rmsds_dict = {
        "4W5N_self": rmsd_to_ref(gen_a_aligned, ref_a),
        "4W5N_to_9K6T": rmsd_to_ref(gen_a_aligned, ref_b),
        "9K6T_to_4W5N": rmsd_to_ref(gen_b_aligned, ref_a),
        "9K6T_self": rmsd_to_ref(gen_b_aligned, ref_b),
    }
    plot_rmsd_distributions(rmsds_dict, os.path.join(out_dir, "rmsd_distributions.png"))

    # B. 结构域拆分
    domains_dict = {
        "4W5N→4W5N": domain_rmsd_analysis(gen_a_aligned, ref_a, common_res),
        "4W5N→9K6T": domain_rmsd_analysis(gen_a_aligned, ref_b, common_res),
        "9K6T→4W5N": domain_rmsd_analysis(gen_b_aligned, ref_a, common_res),
        "9K6T→9K6T": domain_rmsd_analysis(gen_b_aligned, ref_b, common_res),
    }
    plot_domain_rmsd(domains_dict, os.path.join(out_dir, "domain_rmsd_barplot.png"))

    # C. 接触频率差异
    freq_a = contact_frequency(gen_a_aligned)
    freq_b = contact_frequency(gen_b_aligned)
    plot_contact_diff(freq_a, freq_b, os.path.join(out_dir, "contact_freq_diff.png"),
                      label_a=args.pdb_a, label_b=args.pdb_b)

    # D. 零假设对比
    null_gen = null_baseline(ref_a, n_samples=gen_a_aligned.shape[0], noise_std=2.0)
    null_rmsd = rmsd_to_ref(null_gen, ref_a)
    print(f"\n[零假设] 高斯噪声 (σ=2Å) RMSD: mean={null_rmsd.mean():.2f}Å, "
          f"min={null_rmsd.min():.2f}Å")
    print(f"[对比] DynaFlow 4W5N→4W5N: mean={rmsds_dict['4W5N_self'].mean():.2f}Å, "
          f"min={rmsds_dict['4W5N_self'].min():.2f}Å")

    # 汇总表格
    summary = {
        "metric": ["自 RMSD mean", "自 RMSD min", "跨结构 RMSD mean", "跨结构 RMSD min"],
        f"{args.pdb_a}_gen": [
            rmsds_dict["4W5N_self"].mean(),
            rmsds_dict["4W5N_self"].min(),
            rmsds_dict["4W5N_to_9K6T"].mean(),
            rmsds_dict["4W5N_to_9K6T"].min(),
        ],
        f"{args.pdb_b}_gen": [
            rmsds_dict["9K6T_self"].mean(),
            rmsds_dict["9K6T_self"].min(),
            rmsds_dict["9K6T_to_4W5N"].mean(),
            rmsds_dict["9K6T_to_4W5N"].min(),
        ],
    }
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(out_dir, "state_analysis_summary.csv"), index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n写入 -> {out_dir}")


if __name__ == "__main__":
    main()
