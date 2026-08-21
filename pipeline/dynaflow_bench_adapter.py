#!/usr/bin/env python3
"""dynaflow_bench_adapter.py — 把 BioEmu 官方 benchmark 接入 RPDynaFlow 结构条件采样。

三个子命令：
  --featurize            把 cond_structs/*/<tc>_cond.pdb 特征化为 data/static/<TC>.npz
  --sample [--benchmark] 对每个 test case 调 gen_ensembles 采样（跳过已有 npz，可断点续传）
  --to-xtc   [--benchmark] 把采样 npz（gen，Å）转成官方样本格式 samples.xtc + topology.pdb

样本数默认 4000/序列（官方推荐）。采样为长时间 GPU 任务，按 benchmark 分批后台执行。

Run:
  python3 dynaflow_bench_adapter.py --featurize
  python3 dynaflow_bench_adapter.py --sample --benchmark ood60
  python3 dynaflow_bench_adapter.py --to-xtc --benchmark ood60
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))

BENCH_ROOT = os.path.join(RESULTS, "ago2", "bioemu_bench")
COND_DIR = os.path.join(BENCH_ROOT, "cond_structs")
RAW_DIR = os.path.join(BENCH_ROOT, "samples_raw")      # 采样 npz
SAMPLE_DIR = os.path.join(BENCH_ROOT, "samples")        # 官方 xtc+pdb
STATIC_DIR = os.path.join(DATA, "static")

CKPT_DEFAULT = os.path.join(RESULTS, "checkpoints", "flow_model_r15.pt")

BENCHMARKS = ["ood60", "oodval", "domainmotion", "localunfolding",
              "crypticpocket", "folding_free_energies", "md_emulation"]


def load_seq_map():
    return pd.read_csv(os.path.join(BENCH_ROOT, "sequence_map.csv"))


def ok_test_cases(benchmark=None):
    df = load_seq_map()
    df = df[df["status"] == "ok"]
    if benchmark:
        df = df[df["benchmark"] == benchmark]
    return df


def cond_pdb_path(tc, benchmark):
    return os.path.join(COND_DIR, benchmark, f"{tc}_cond.pdb")


def pid_of(tc):
    return tc.upper()


def featurize_all():
    df = ok_test_cases()
    os.makedirs(STATIC_DIR, exist_ok=True)
    sys.path.insert(0, HERE)
    from featurize_static_pdb import write_static_npz
    done = 0
    for _, r in df.iterrows():
        pid = pid_of(r["test_case"])
        cond = cond_pdb_path(r["test_case"], r["benchmark"])
        if not os.path.exists(cond):
            print(f"[skip] 缺少条件结构 {cond}")
            continue
        out = write_static_npz(pid, cond, STATIC_DIR)
        done += 1
    print(f"[featurize] {done} 个条件结构 -> {STATIC_DIR}")


def sample_benchmark(benchmark, ckpt, n_gen, steps):
    df = ok_test_cases(benchmark)
    os.makedirs(RAW_DIR, exist_ok=True)
    out_dir = os.path.join(RAW_DIR, benchmark)
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for _, r in df.iterrows():
        pid = pid_of(r["test_case"])
        npz = os.path.join(out_dir, f"{pid}.npz")
        if os.path.exists(npz) and os.path.getsize(npz) > 1000:
            continue
        todo.append(r["test_case"])
    print(f"[sample] {benchmark}: {len(df)-len(todo)}/{len(df)} 已就绪，待采样 {len(todo)}")
    for tc in todo:
        pid = pid_of(tc)
        print(f"[sample] {pid} n_gen={n_gen}")
        cmd = [
            sys.executable, os.path.join(HERE, "gen_ensembles.py"),
            "--ckpt", ckpt, "--systems", pid,
            "--static", "--n-gen", str(n_gen), "--steps", str(steps),
            "--no-pdb", "--out", out_dir,
        ]
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"[fail] {pid} 采样返回 {r.returncode}")


def to_xtc(benchmark):
    import numpy as np
    import mdtraj
    df = ok_test_cases(benchmark)
    raw_dir = os.path.join(RAW_DIR, benchmark)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    done = 0
    for _, r in df.iterrows():
        tc = r["test_case"]
        pid = pid_of(tc)
        npz = os.path.join(raw_dir, f"{pid}.npz")
        cond = cond_pdb_path(tc, benchmark)
        if not os.path.exists(npz) or not os.path.exists(cond):
            print(f"[skip] 缺 npz 或 cond: {tc}")
            continue
        z = np.load(npz, allow_pickle=True)
        gen = z["gen"].astype(np.float64)          # (S,A,3) Å
        top = mdtraj.load_pdb(cond)
        if gen.shape[1] != top.n_atoms:
            print(f"[skip] 原子数不匹配 {tc}: gen {gen.shape[1]} vs top {top.n_atoms}")
            continue
        sub = os.path.join(SAMPLE_DIR, benchmark, tc)
        os.makedirs(sub, exist_ok=True)
        traj = mdtraj.Trajectory(gen / 10.0, top.topology)  # Å -> nm
        traj.save_xtc(os.path.join(sub, "samples.xtc"))
        top.save_pdb(os.path.join(sub, "topology.pdb"))
        done += 1
    print(f"[to-xtc] {benchmark}: {done} 个 test case -> {SAMPLE_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featurize", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--to-xtc", action="store_true")
    ap.add_argument("--benchmark", default="")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--n-gen", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    if args.featurize:
        featurize_all()
    elif args.sample:
        if not args.benchmark:
            ap.error("--sample 需要 --benchmark")
        sample_benchmark(args.benchmark, args.ckpt, args.n_gen, args.steps)
    elif args.to_xtc:
        if not args.benchmark:
            ap.error("--to-xtc 需要 --benchmark")
        to_xtc(args.benchmark)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
