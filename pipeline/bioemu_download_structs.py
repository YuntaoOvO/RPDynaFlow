#!/usr/bin/env python3
"""bioemu_download_structs.py — 为 BioEmu 官方 benchmark 准备 RPDynaFlow 条件结构。

RPDynaFlow 是结构条件采样模型（从单一静态结构采样构象集合），而 BioEmu 官方
benchmark 是"从序列采样"。本脚本把官方 benchmark 的每个 test case 映射到一个
条件结构（单体蛋白，全原子，单链），供 featurize_static_pdb + gen_ensembles 使用。

三组来源：
  1. multiconf 5 类（ood60/oodval/domainmotion/localunfolding/crypticpocket）：
     读 references.csv 的 pdbidchain_i（第一个参考构象），从 RCSB 下载全原子
     PDB，提取目标链。
  2. folding_free_energies：野生型（system_info.csv mutant=False）的
     reference_wildtypes/<name>.pdb（已存在，不下载）。
  3. md_emulation：从 test_case（cath1_<PDBid><chain><domain>）解析 PDB id 与链，
     RCSB 下载后按 test_case 序列在链序列中定位域片段。

输出：
  <out>/cond_structs/<benchmark>/<test_case>_cond.pdb
  <out>/sequence_map.csv    # 每个 test case 的条件结构与序列关系

Run:
  python3 bioemu_download_structs.py
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))

# 官方 benchmark 数据资产（只读）
# clone https://github.com/microsoft/bioemu-benchmarks to $ROOT/bioemu-benchmarks
# or point BIOEMU_BENCH_DIR at it
BENCH_DIR = os.environ.get(
    "BIOEMU_BENCH_DIR",
    os.path.join(os.path.dirname(HERE), "bioemu-benchmarks", "bioemu_benchmarks", "assets"))
MULTICONF_DIR = os.path.join(BENCH_DIR, "multiconf_benchmark_0.1")
FOLDING_DIR = os.path.join(BENCH_DIR, "folding_free_energies_benchmark_0.1", "folding_free_energies")
MDEMU_DIR = os.path.join(BENCH_DIR, "md_emulation_benchmark_0.1", "md_emulation")

RCSB_BASE = "https://files.rcsb.org/download/{}.pdb"

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # 常见修饰残基（保留，元素/残基类型由 featurize 的 RESTYPE 兜底到 23）
    "MSE", "SEC", "PYL", "HYP", "SEP", "TPO", "PTR", "CSO", "CME", "MLY",
    "KCX", "LLP", "PCA", "FME", "ASX", "GLX", "UNK",
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B",
    "GLX": "Z", "UNK": "X",
}


def parse_pdbidchain(s):
    """'7B5Y_A' -> ('7B5Y', 'A')."""
    pdbid, _, chain = s.partition("_")
    return pdbid.upper(), chain


def download_pdb(pdbid, cache_dir):
    """下载 RCSB PDB（缓存），返回本地路径。"""
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"{pdbid.lower()}.pdb")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    import urllib.request
    url = RCSB_BASE.format(pdbid.lower())
    for attempt in range(2):
        try:
            urllib.request.urlretrieve(url, p)
            if os.path.getsize(p) > 1000:
                return p
        except Exception as e:
            print(f"  [retry {attempt+1}] download {pdbid}: {e}")
            time.sleep(1)
    print(f"  [skip] {pdbid} download failed (obsolete/404)")
    return None


def iter_atom_records(pdb_path, chain_id, protein_only=True):
    """逐行解析 PDB，产出指定链的 ATOM 记录（标准氨基酸，去 HETATM/水）。"""
    recs = []
    seen_model = False
    with open(pdb_path, errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                if seen_model:
                    break  # 只取第一个 MODEL（NMR 多构象）
                seen_model = True
                continue
            if rec == "ENDMDL":
                if seen_model:
                    break
                continue
            if rec != "ATOM":
                continue
            chain = line[21] if len(line) > 21 else " "
            resn = line[17:20].strip().upper()
            if chain_id is not None and chain != chain_id:
                continue
            if protein_only and resn not in STANDARD_AA:
                continue
            recs.append(line.rstrip("\n"))
    return recs


def chain_sequence(recs):
    """从 ATOM 记录列表得到残基序列（按 resseq 去重、升序）。"""
    seen, order = [], {}
    for line in recs:
        resseq = line[22:26].strip()
        resn = line[17:20].strip().upper()
        key = resseq
        if key not in order:
            order[key] = len(order)
            seen.append(resn)
    return "".join(AA3_TO_1.get(r, "X") for r in seen)


def write_cond_pdb(recs, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("HEADER    RPDYNAFLOW CONDITIONING STRUCTURE\n")
        for line in recs:
            fh.write(line + "\n")
        fh.write("END\n")


def dominant_chain(pdb_path):
    """对未指定链的 PDB，返回残基数最多的蛋白链 id。"""
    chains = {}
    with open(pdb_path, errors="ignore") as fh:
        for line in fh:
            if line[:6].strip() != "ATOM":
                continue
            resn = line[17:20].strip().upper()
            if resn not in STANDARD_AA:
                continue
            chain = line[21]
            chains.setdefault(chain, 0)
            chains[chain] += 1
    if not chains:
        return None
    return max(chains, key=chains.get)


def locate_subsequence(seq, sub):
    """在 seq 中找 sub 的连续子串起始索引（精确匹配），失败返回 None。"""
    idx = seq.find(sub)
    return idx if idx >= 0 else None


def process_multiconf(benchmark, out_root, cache_dir):
    rows = []
    ref_csv = os.path.join(MULTICONF_DIR, benchmark, "references.csv")
    if not os.path.exists(ref_csv):
        return rows
    df = pd.read_csv(ref_csv)
    cond_col = "pdbidchain_i" if "pdbidchain_i" in df.columns else "pdbidchain"
    j_col = "pdbidchain_j" if "pdbidchain_j" in df.columns else None
    ref_dir = os.path.join(MULTICONF_DIR, benchmark, "reference")
    for _, r in df.iterrows():
        tc = r["test_case"]
        pdbid, chain = parse_pdbidchain(r[cond_col])
        # 该 test case 的参考构象数（evaluate_test_case 要求 >1）
        ref_pdbs = [f for f in os.listdir(os.path.join(ref_dir, tc))
                    if f.endswith(".pdb")] if os.path.isdir(os.path.join(ref_dir, tc)) else []
        n_ref = len(ref_pdbs)
        pdb_path = download_pdb(pdbid, cache_dir)
        # 下载失败：回退到第二个参考构象（同一蛋白的另一构象）
        if pdb_path is None and j_col and pd.notna(r.get(j_col)):
            pdbid2, chain2 = parse_pdbidchain(r[j_col])
            pdb_path = download_pdb(pdbid2, cache_dir)
            if pdb_path is not None:
                pdbid, chain = pdbid2, chain2
        if pdb_path is None:
            rows.append(dict(benchmark=benchmark, test_case=tc, cond=pdbid + "_" + chain,
                             n_ref=n_ref, n_res=0, status="download-failed"))
            continue
        recs = iter_atom_records(pdb_path, chain)
        # 无 ATOM（链标注与 RCSB 不符）：回退到最大蛋白链
        if not recs:
            alt_chain = dominant_chain(pdb_path)
            if alt_chain is not None and alt_chain != chain:
                recs = iter_atom_records(pdb_path, alt_chain)
                if recs:
                    chain = alt_chain
        if not recs:
            rows.append(dict(benchmark=benchmark, test_case=tc, cond=pdbid + "_" + chain,
                             n_ref=n_ref, n_res=0, status="no-atoms"))
            continue
        seq = chain_sequence(recs)
        out = os.path.join(out_root, benchmark, f"{tc}_cond.pdb")
        write_cond_pdb(recs, out)
        rows.append(dict(benchmark=benchmark, test_case=tc, cond=pdbid + "_" + chain,
                         n_ref=n_ref, n_res=len(recs), seq_len=len(seq), status="ok"))
    return rows


def process_folding(out_root):
    rows = []
    info = pd.read_csv(os.path.join(FOLDING_DIR, "system_info.csv"))
    wt = info[~info.mutant]  # 野生型
    ref_dir = os.path.join(FOLDING_DIR, "reference_wildtypes")
    for _, r in wt.iterrows():
        name = r["name"]
        pdb_path = os.path.join(ref_dir, f"{name}.pdb")
        if not os.path.exists(pdb_path):
            rows.append(dict(benchmark="folding_free_energies", test_case=name, cond=name,
                             n_ref=1, n_res=0, status="missing-wt-pdb"))
            continue
        chain = dominant_chain(pdb_path)
        recs = iter_atom_records(pdb_path, chain)
        if not recs:
            rows.append(dict(benchmark="folding_free_energies", test_case=name, cond=name,
                             n_ref=1, n_res=0, status="no-atoms"))
            continue
        out = os.path.join(out_root, "folding_free_energies", f"{name}_cond.pdb")
        write_cond_pdb(recs, out)
        rows.append(dict(benchmark="folding_free_energies", test_case=name, cond=name,
                         n_ref=1, n_res=len(recs), seq_len=len(chain_sequence(recs)),
                         status="ok"))
    return rows


def process_mdemulation(out_root, cache_dir):
    rows = []
    tc = pd.read_csv(os.path.join(MDEMU_DIR, "testcases.csv"))
    for _, r in tc.iterrows():
        test_case = r["test_case"]      # cath1_1bl0A02
        target_seq = r["sequence"]
        # 解析：cath1_ 前缀 + 4 位 PDB id（小写）+ 1 位链（大写）+ 2 位域号
        body = test_case.split("_", 1)[1]   # 1bl0A02
        pdbid = body[:4].upper()
        chain = body[4]
        pdb_path = download_pdb(pdbid, cache_dir)
        if pdb_path is None:
            rows.append(dict(benchmark="md_emulation", test_case=test_case,
                             cond=pdbid + "_" + chain, n_ref=1, n_res=0,
                             seq_len=len(target_seq), status="download-failed"))
            continue
        recs_all = iter_atom_records(pdb_path, chain)
        seq_all = chain_sequence(recs_all)
        idx = locate_subsequence(seq_all, target_seq)
        if idx is None:
            rows.append(dict(benchmark="md_emulation", test_case=test_case,
                             cond=pdbid + "_" + chain, n_ref=1, n_res=0,
                             seq_len=len(seq_all), status="seq-not-located"))
            continue
        # 按序列定位域片段：保留 target_seq 长度对应的残基
        # 残基边界用 resseq 排序后按 idx 切片
        recs = recs_all
        # 重建残基级别的记录，按 resseq 升序
        resid_order = []
        for line in recs:
            rs = line[22:26].strip()
            if rs not in [x[0] for x in resid_order]:
                resid_order.append((rs, line[17:20].strip().upper()))
        # resid_order 已按出现顺序，序列按此顺序
        seg = resid_order[idx:idx + len(target_seq)]
        seg_rs = {x[0] for x in seg}
        recs_seg = [line for line in recs if line[22:26].strip() in seg_rs]
        out = os.path.join(out_root, "md_emulation", f"{test_case}_cond.pdb")
        write_cond_pdb(recs_seg, out)
        rows.append(dict(benchmark="md_emulation", test_case=test_case,
                         cond=pdbid + "_" + chain, n_ref=1, n_res=len(recs_seg),
                         seq_len=len(target_seq), status="ok"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "ago2", "bioemu_bench"))
    ap.add_argument("--cache", default=os.path.join(RESULTS, "ago2", "bioemu_bench", "rcsb_cache"))
    ap.add_argument("--benchmarks", default="ood60,oodval,domainmotion,localunfolding,crypticpocket,folding_free_energies,md_emulation")
    args = ap.parse_args()

    out_root = os.path.join(args.out, "cond_structs")
    os.makedirs(out_root, exist_ok=True)
    cache_dir = args.cache

    all_rows = []
    for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
        print(f"[下载] {b}")
        if b in ("ood60", "oodval", "domainmotion", "localunfolding", "crypticpocket"):
            rows = process_multiconf(b, out_root, cache_dir)
        elif b == "folding_free_energies":
            rows = process_folding(out_root)
        elif b == "md_emulation":
            rows = process_mdemulation(out_root, cache_dir)
        else:
            print(f"  unknown benchmark {b}")
            continue
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"  {ok}/{len(rows)} 条件结构就绪")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.out, "sequence_map.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[汇总] {len(df)} test case -> {out_csv}")
    print(df["status"].value_counts().to_string())
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
