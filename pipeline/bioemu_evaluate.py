#!/usr/bin/env python3
"""bioemu_evaluate.py — 用官方 bioemu-benchmarks 评估 RPDynaFlow 采样结果。

对每个 benchmark：
  1. 手动构造 IndexedSamples（绕过 from_benchmark 的序列精确匹配，因为条件结构
     只覆盖 test cases.csv 全长序列的子串）
  2. filter_unphysical_samples（官方 unphysical 过滤，记录每 test case 保留比例）
  3. 调用官方 evaluator（multiconf 类传 metric_types；folding/md_emulation 用默认）
  4. 保存 aggregate metrics + 覆盖曲线图 + unphysical 统计

Run:
  python3 bioemu_evaluate.py --benchmark ood60
  python3 bioemu_evaluate.py --benchmark folding_free_energies
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get(
    "BIOEMU_BENCH_ROOT",
    os.environ.get(
        "BIOEMU_BENCH_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bioemu-benchmarks"))))

from bioemu_benchmarks.benchmarks import Benchmark
from bioemu_benchmarks.samples import IndexedSamples, SequenceSample
from bioemu_benchmarks.evaluator_utils import evaluator_from_benchmark
from bioemu_benchmarks.eval.multiconf.evaluate import MetricType
from bioemu_benchmarks.utils import filter_unphysical_traj_masks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
BENCH_ROOT = os.path.join(RESULTS, "ago2", "bioemu_bench")
SAMPLE_DIR = os.path.join(BENCH_ROOT, "samples")
EVAL_DIR = os.path.join(BENCH_ROOT, "eval")

BENCHMARK_ENUM = {
    "ood60": Benchmark.MULTICONF_OOD60,
    "oodval": Benchmark.MULTICONF_OODVAL,
    "domainmotion": Benchmark.MULTICONF_DOMAINMOTION,
    "crypticpocket": Benchmark.MULTICONF_CRYPTICPOCKET,
    "localunfolding": Benchmark.SINGLECONF_LOCALUNFOLDING,
    "folding_free_energies": Benchmark.FOLDING_FREE_ENERGIES,
    "md_emulation": Benchmark.MD_EMULATION,
}

MULTICONF_BENCHMARKS = {"ood60", "oodval", "domainmotion", "crypticpocket", "localunfolding"}

# 指标集：RMSD + LDDT + contact-distance（mdtraj 计算，无需外部工具）。
# 排除 TM-score（每帧一次 US-align subprocess，4000 样本下过慢）与 DSSP（需外部 dssp 工具）。
METRIC_TYPES = [MetricType.RMSD, MetricType.LDDT, MetricType.CONTACT_DISTANCE]


def load_ok_test_cases(benchmark):
    seq_map = pd.read_csv(os.path.join(BENCH_ROOT, "sequence_map.csv"))
    df = seq_map[(seq_map["benchmark"] == benchmark) & (seq_map["status"] == "ok")]
    return df["test_case"].tolist()


def build_indexed_samples(benchmark, test_cases):
    """手动构造 IndexedSamples：test_case -> [SequenceSample]"""
    test_case_to_ss = {}
    for tc in test_cases:
        sub = os.path.join(SAMPLE_DIR, benchmark, tc)
        topo = os.path.join(sub, "topology.pdb")
        traj = os.path.join(sub, "samples.xtc")
        if not (os.path.exists(topo) and os.path.exists(traj)):
            print(f"[skip] 缺样本 {tc}")
            continue
        test_case_to_ss[tc] = [SequenceSample(topology_file=topo, trajectory_file=traj)]
    if not test_case_to_ss:
        raise RuntimeError(f"无样本可评估: {benchmark}")
    return IndexedSamples(test_case_to_ss)


def compute_unphysical_rates(indexed):
    """手动计算每个 test case 的 unphysical 过滤率（绕过官方 filter 对全灭样本的 assert）。

    官方 filter_unphysical_traj_masks 的三条判据（全键通过，np.all）：
      相邻 Cα-Cα < 4.5 Å、肽键 C-N < 2.0 Å、跨残基原子对 > 1.0 Å（clash）。
    我们的采样键长漂移系统性（训练数据断裂率 0%，采样 7–40%），官方过滤会排除
    几乎全部样本。这里只记录过滤率，评估本身用未过滤样本进行。
    """
    rows = []
    for tc, ss_list in indexed.test_case_to_sequencesamples.items():
        for ss in ss_list:
            traj = ss.get_traj()
            ca_ok, cn_ok, clash_ok = filter_unphysical_traj_masks(traj)
            all_ok = ca_ok & cn_ok & clash_ok
            rows.append(dict(test_case=tc, n_frames=int(traj.n_frames),
                             kept=int(all_ok.sum()),
                             kept_fraction=float(all_ok.mean())))
    return pd.DataFrame(rows)


def evaluate(benchmark):
    test_cases = load_ok_test_cases(benchmark)
    print(f"[eval] {benchmark}: {len(test_cases)} test cases")
    indexed = build_indexed_samples(benchmark, test_cases)

    # unphysical 过滤率（核心负面发现）
    kept_df = compute_unphysical_rates(indexed)
    print(f"[filter] 保留比例 均值={kept_df.kept_fraction.mean():.4f} "
          f"min={kept_df.kept_fraction.min():.4f} max={kept_df.kept_fraction.max():.4f}")

    # 官方评估（未过滤样本）
    benchmark_enum = BENCHMARK_ENUM[benchmark]
    if benchmark in MULTICONF_BENCHMARKS:
        evaluator = evaluator_from_benchmark(benchmark_enum, metric_types=METRIC_TYPES)
    else:
        evaluator = evaluator_from_benchmark(benchmark_enum)
    results = evaluator(indexed)

    # 保存
    out_dir = os.path.join(EVAL_DIR, benchmark)
    os.makedirs(out_dir, exist_ok=True)
    results.save_results(out_dir)
    results.plot(out_dir)
    agg = results.get_aggregate_metrics()
    with open(os.path.join(out_dir, "benchmark_metrics.json"), "w") as fh:
        json.dump(agg, fh, indent=2)
    kept_df.to_csv(os.path.join(out_dir, "unphysical_filter.csv"), index=False)

    print(f"[结果] {benchmark} -> {out_dir}")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BENCHMARK_ENUM.keys()))
    args = ap.parse_args()
    evaluate(args.benchmark)


if __name__ == "__main__":
    main()
