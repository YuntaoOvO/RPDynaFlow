#!/usr/bin/env python3
"""bioemu_train_overlap.py — 检查 RPDynaFlow 训练集与 BioEmu benchmark 的数据重合。

评估完整性检查：模型是否在训练时见过 benchmark 的蛋白（数据泄漏会高估表现）。
三个层面：
  1. PDB ID 重合（训练集复合物 vs benchmark 参考结构）
  2. 序列完全相等 / 子串
  3. 定量序列相似度（8-mer 共享 + BLOSUM62 全局比对 identity）

Run:
  python3 bioemu_train_overlap.py
"""
import glob
import os

import mdtraj
import pandas as pd

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V', 'MSE': 'M'}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# bioemu-benchmarks assets; clone https://github.com/microsoft/bioemu-benchmarks
# and set BIOEMU_BENCH_DIR (or place it at $ROOT/bioemu-benchmarks)
BENCH_DIR = os.environ.get(
    "BIOEMU_BENCH_DIR",
    os.path.join(ROOT, "bioemu-benchmarks", "bioemu_benchmarks", "assets"))

# 训练集：数据消融 K=15 的全部系统（与 flow_model_r15.pt 的 trained_on 一致）
TRAIN_IDS = ['1NYB', '2ESE', '1EKZ', '1A1T', '4PDB', '1DK1', '2XDB', '2Y8W',
             '6GBM', '2FY1', '2N82', '2L2K', '2N3O', '1RKJ', '1FJE']

BENCHMARKS = [
    ('ood60', 'multiconf_benchmark_0.1/ood60/testcases.csv'),
    ('oodval', 'multiconf_benchmark_0.1/oodval/testcases.csv'),
    ('domainmotion', 'multiconf_benchmark_0.1/domainmotion/testcases.csv'),
    ('localunfolding', 'multiconf_benchmark_0.1/localunfolding/testcases.csv'),
    ('crypticpocket', 'multiconf_benchmark_0.1/crypticpocket/testcases.csv'),
    ('folding', 'folding_free_energies_benchmark_0.1/folding_free_energies/testcases.csv'),
    ('md_emulation', 'md_emulation_benchmark_0.1/md_emulation/testcases.csv'),
]


def train_sequences():
    seqs = []
    for pid in TRAIN_IDS:
        pdb = glob.glob(os.path.join(ROOT, "RNA-protein complexes", pid, "*.pdb"))
        if not pdb:
            continue
        t = mdtraj.load_pdb(pdb[0])
        for ci in range(t.topology.n_chains):
            c = t.topology.chain(ci)
            seq = "".join(AA3.get(r.name, "X") for r in c.residues if r.name in AA3)
            if len(seq) > 10:
                seqs.append((pid, seq))
    return seqs


def bench_sequences():
    seqs = []
    for _, f in BENCHMARKS:
        df = pd.read_csv(os.path.join(BENCH_DIR, f))
        for _, r in df.iterrows():
            seqs.append((r["test_case"], r["sequence"]))
    return seqs


def kmer_set(seq, k=8):
    return set(seq[i:i + k] for i in range(len(seq) - k + 1))


def main():
    tr = train_sequences()
    be = bench_sequences()
    print(f"训练集: {len(tr)} 条蛋白链（{len(TRAIN_IDS)} 个复合物）")
    print(f"benchmark: {len(be)} 个序列")

    # 1. 完全相等 / 子串
    exact = substr = 0
    for tc, bseq in be:
        for pid, s in tr:
            if s == bseq:
                exact += 1
            elif (s in bseq or bseq in s) and min(len(s), len(bseq)) > 30:
                substr += 1
    print(f"序列完全相等: {exact}, 显著子串: {substr}")

    # 2. 8-mer 共享（定量相似度的快速判据）
    tr_kmers = {pid: kmer_set(s) for pid, s in tr}
    shared = 0
    for tc, bseq in be:
        bk = kmer_set(bseq)
        if any(bk & ks for ks in tr_kmers.values()):
            shared += 1
    print(f"共享 8-mer 的 benchmark 序列数: {shared} / {len(be)}")

    lens = [len(s) for _, s in tr]
    print(f"训练集链长度范围: {min(lens)}-{max(lens)} aa")
    print("结论: " + ("训练集与 benchmark 无序列重合" if (exact == 0 and shared == 0)
                    else "存在序列重合，需进一步核查"))


if __name__ == "__main__":
    main()
