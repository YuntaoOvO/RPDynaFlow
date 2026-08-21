#!/usr/bin/env python3
"""compare_bioemu.py — BioEmu 基线对比（同 4W5N/9K6T 系统）

对比 DynaFlow 与 BioEmu 在相同系统上的 Coverage/Precision 指标。
需要：
  - DynaFlow 采样结果：results/samples/ago2/r15/{4W5N,9K6T}.npz
  - BioEmu 采样结果：data/bioemu_samples/{4W5N,9K6T}.npz（需手动运行 BioEmu 生成）

评估协议：
  - Coverage@3Å/5Å: BioEmu-style 单结构覆盖
  - Precision: 假设 MD 轨迹为 ground-truth（如果可用），或使用交叉验证
  - FNC (Foldedness/Nonfoldedness/Collapse): 需要 Rg 计算

Run:
  python3 compare_bioemu.py --dynaflow ../results/samples/ago2/r15 \
                             --bioemu ../data/bioemu_samples
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
from ago2_multiconf_eval import extract_ca_map, common_residues, ca_vectors
from featurize_static_pdb import _find_pdb

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
COV_THRESHOLDS = np.arange(0.5, 12.01, 0.25)


def _kabsch(mobile, ref):
    mc, rc = mobile.mean(0), ref.mean(0)
    H = (mobile - mc).T @ (ref - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (mobile - mc) @ R.T + rc


def rmsd_aligned(mobile, ref):
    out = np.empty(len(mobile), dtype=np.float64)
    for i in range(len(mobile)):
        aligned = _kabsch(mobile[i], ref)
        out[i] = np.sqrt(((aligned - ref) ** 2).sum(-1).mean())
    return out


def coverage_at_thresholds(gen_rmsd_to_ref, thresholds=COV_THRESHOLDS, min_frac=0.001):
    n = len(gen_rmsd_to_ref)
    k = max(1, int(np.ceil(min_frac * n)))
    dk = np.partition(gen_rmsd_to_ref, k - 1)[k - 1]
    return np.array([float(dk < t) for t in thresholds])


def load_ca_from_npz(npz_path, ref_pdb):
    """提取 CA 坐标并对齐到参考结构残基"""
    from featurize_static_pdb import parse_static_pdb
    from importlib import import_module
    RESTYPE = import_module("05_postprocess").RESTYPE

    z = np.load(npz_path, allow_pickle=True)
    gen = z["gen"] if "gen" in z else z["coords"]
    src = str(z.get("source_pdb", ""))
    if not src or not os.path.exists(src):
        pid = os.path.basename(npz_path).replace(".npz", "")
        src = _find_pdb(pid)

    _, _, _, res_rt, _, atoms = parse_static_pdb(src)
    ca_idx = [i for i, a in enumerate(atoms) if a["name"] == "CA"
              and RESTYPE.get(a["resn"], 24) < 20]
    resnums = [int(atoms[i]["resseq"]) for i in ca_idx]

    ca_map_ref = extract_ca_map(ref_pdb)
    common_res = sorted(set(resnums) & set(ca_map_ref.keys()))
    idx = [i for i, r in enumerate(resnums) if r in common_res]
    ref_ca = ca_vectors(ca_map_ref, common_res)
    gen_ca = gen[:, idx]
    return gen_ca, ref_ca, common_res


def eval_method(npz_path, ref_pdb, label):
    """单方法评估：Coverage 曲线 + RMSD 统计"""
    gen_ca, ref_ca, common_res = load_ca_from_npz(npz_path, ref_pdb)
    rmsds = rmsd_aligned(gen_ca, ref_ca)
    cov = coverage_at_thresholds(rmsds)

    i3 = int(np.where(np.isclose(COV_THRESHOLDS, 3.0))[0][0])
    i5 = int(np.where(np.isclose(COV_THRESHOLDS, 5.0))[0][0])

    return {
        "label": label,
        "n_samples": int(len(gen_ca)),
        "n_ca": len(common_res),
        "rmsd_mean": float(rmsds.mean()),
        "rmsd_min": float(rmsds.min()),
        "rmsd_p10": float(np.percentile(rmsds, 10)),
        "cov_3A": float(cov[i3]),
        "cov_5A": float(cov[i5]),
        "cov_auc": float((getattr(np, "trapezoid", None) or np.trapz)(cov, COV_THRESHOLDS) / (COV_THRESHOLDS[-1] - COV_THRESHOLDS[0])),
        "coverage_curve": cov,
    }


def plot_comparison(results, out_path, title="DynaFlow vs BioEmu"):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"DynaFlow": "#3C5A78", "BioEmu": "#D97D54"}
    for r in results:
        label = r["label"]
        color = colors.get(label.split()[0], "gray")
        ax.plot(COV_THRESHOLDS, r["coverage_curve"], label=label, lw=2.5, color=color)

    ax.set_xlabel("RMSD threshold (Å)", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图] 对比曲线: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynaflow", required=True, help="DynaFlow samples dir")
    ap.add_argument("--bioemu", required=True, help="BioEmu samples dir")
    ap.add_argument("--out", default="", help="default: results/comparison/")
    ap.add_argument("--systems", default="4W5N,9K6T")
    args = ap.parse_args()

    systems = [s.strip().upper() for s in args.systems.split(",") if s.strip()]
    out_dir = args.out or os.path.join(RESULTS, "comparison", "bioemu")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for pid in systems:
        pdb = _find_pdb(pid)
        if not pdb:
            print(f"[SKIP] PDB 不存在: {pid}")
            continue

        npz_df = os.path.join(args.dynaflow, f"{pid}.npz")
        npz_be = os.path.join(args.bioemu, f"{pid}.npz")

        if not os.path.exists(npz_df):
            print(f"[SKIP] DynaFlow 样本缺失: {npz_df}")
            continue
        if not os.path.exists(npz_be):
            print(f"[WARN] BioEmu 样本缺失: {npz_be}")
            print(f"       需手动运行 BioEmu 生成 {pid} 系综")
            continue

        print(f"\n[评估] {pid}")
        r_df = eval_method(npz_df, pdb, f"DynaFlow {pid}")
        r_be = eval_method(npz_be, pdb, f"BioEmu {pid}")

        all_results.extend([r_df, r_be])

        # 单系统对比图
        plot_comparison([r_df, r_be],
                        os.path.join(out_dir, f"coverage_{pid}.png"),
                        title=f"{pid} Coverage: DynaFlow vs BioEmu")

    if not all_results:
        sys.exit("[ERROR] 无有效对比数据")

    # 汇总表格
    rows = [{k: v for k, v in r.items() if k != "coverage_curve"} for r in all_results]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print(f"\n{df.to_string(index=False)}")

    # 全局对比图（所有系统）
    plot_comparison(all_results,
                    os.path.join(out_dir, "coverage_all.png"),
                    title="DynaFlow vs BioEmu (all systems)")

    print(f"\n写入 -> {out_dir}")


if __name__ == "__main__":
    main()
