#!/usr/bin/env python3
"""esmdynamic_overlay.py — compare DynaFlow ensemble contacts to ESMDynamic maps.

ESMDynamic predicts dynamic contacts, contact frequency, and kinetics from sequence
(protein only). This script:
  1. Computes Cα contact frequency / dynamic flags from DynaFlow sample npz files.
  2. Loads ESMDynamic outputs (or runs prediction if model available).
  3. Writes comparison tables and heatmap figures to results/ago2/esmdynamic/.

Run (after ESMDynamic prediction):
  run_esmdynamic --fasta ago2.fasta --output_dir ../results/ago2/esmdynamic/pred --save_txt
  python3 esmdynamic_overlay.py --samples-dir ../results/samples/ago2/r15 \\
      --esm-dir ../results/ago2/esmdynamic/pred

Or DynaFlow-only contact analysis:
  python3 esmdynamic_overlay.py --samples-dir ../results/samples/ago2/r15 --dynaflow-only
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

CONTACT_CUTOFF = 8.0  # Å, ESMDynamic native contact convention


def protein_ca_indices(atoms, res_rt):
  """Index of protein CA atoms in parse order."""
  idx, resnums = [], []
  for i, a in enumerate(atoms):
    if a["name"] != "CA":
      continue
    if RESTYPE.get(a["resn"], 24) >= 20:
      continue
    try:
      resnums.append(int(a["resseq"]))
    except ValueError:
      continue
    idx.append(i)
  return np.array(idx, dtype=np.int64), resnums


def contact_frequency_ca(coords, ca_idx, cutoff=CONTACT_CUTOFF, min_seq_sep=3):
  """coords (F,A,3) -> (L,L) contact frequency for protein CA subset."""
  ca = coords[:, ca_idx]  # (F, L, 3)
  F, L, _ = ca.shape
  # 构建序列分离掩膜（上三角，i < j，j - i >= min_seq_sep）
  i_grid, j_grid = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
  sep_mask = (j_grid - i_grid >= min_seq_sep)

  freq = np.zeros((L, L), dtype=np.float32)
  for f in range(F):
    d = cdist(ca[f], ca[f])  # (L, L)
    contact = (d < cutoff) & (d > 1e-3) & sep_mask
    freq += contact.astype(np.float32)
  freq /= F
  # 对称化
  freq = freq + freq.T
  freq[np.diag_indices(L)] = 0
  return freq


def dynamic_contact_mask(freq, static_freq, switch_thresh=0.2):
  """Pairs that change contact state across ensemble vs static."""
  return (np.abs(freq - static_freq) > switch_thresh) & (
      (freq > 0.1) | (static_freq > 0.1))


def load_esmdynamic_frequency(esm_dir, protein_id, temp_k=320):
  """Load frequency_pred matrix from ESMDynamic output tree."""
  import glob
  pat = os.path.join(esm_dir, protein_id, "frequency",
                     f"*frequency_pred_{temp_k}K.txt")
  hits = glob.glob(pat)
  if not hits:
    pat2 = os.path.join(esm_dir, protein_id, "frequency", f"*_{temp_k}K.txt")
    hits = glob.glob(pat2)
  if not hits:
    return None
  return np.loadtxt(hits[0], dtype=np.float32)


def load_esmdynamic_dynamic_prob(esm_dir, protein_id, temp_k=320):
  import glob
  pat = os.path.join(esm_dir, protein_id, "dynamic", f"*dynamic_prob_{temp_k}K.txt")
  hits = glob.glob(pat)
  if not hits:
    return None
  return np.loadtxt(hits[0], dtype=np.float32)


def analyze_samples(npz_path, pdb_path):
  z = np.load(npz_path, allow_pickle=True)
  gen = z["gen"]
  static = z["static"]
  _, _, _, res_rt, _, atoms = parse_static_pdb(pdb_path)
  ca_idx, resnums = protein_ca_indices(atoms, res_rt)
  if len(ca_idx) < 10:
    raise ValueError(f"too few protein CA in {pdb_path}")
  freq = contact_frequency_ca(gen, ca_idx)
  d0 = np.linalg.norm(static[ca_idx][:, None] - static[ca_idx][None], axis=-1)
  static_freq = ((d0 < CONTACT_CUTOFF) & (d0 > 1e-3)).astype(np.float32)
  dyn_mask = dynamic_contact_mask(freq, static_freq)
  return dict(
    pid=os.path.basename(npz_path).replace(".npz", ""),
    n_ca=len(ca_idx),
    resnums=resnums,
    freq=freq,
    static_freq=static_freq,
    dynamic_mask=dyn_mask,
    n_dynamic_pairs=int(dyn_mask.sum() // 2),
    mean_freq=float(freq[np.triu_indices(len(ca_idx), k=3)].mean()),
  )


def plot_contact_map(mat, title, out_path, vmax=1.0):
  fig, ax = plt.subplots(figsize=(7, 6))
  im = ax.imshow(mat, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
  ax.set_title(title)
  plt.colorbar(im, ax=ax, fraction=0.046)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def esm_indices(resnums, n_esm):
  """Map PDB residue order to 0-based ESMDynamic sequence indices.

  The ESM input sequence is exactly the PDB protein residues in file order, so
  the k-th protein CA residue maps to ESM index k (no resseq arithmetic — that
  would break for 4W5N's 22..859 vs 9K6T's 410..859 numbering).  Returns
  (valid_bool, seq_idx) over the protein CA residue list; valid selects residues
  that fall inside the ESM matrix (shape mismatch -> empty).
  """
  n_ca = len(resnums)
  if n_ca > n_esm:          # DynaFlow has more residues than the ESM sequence
    return np.zeros(n_ca, dtype=bool), np.array([], dtype=np.int64)
  return np.ones(n_ca, dtype=bool), np.arange(n_ca, dtype=np.int64)


def compare_to_esm(dyn, esm_freq, esm_dyn):
  """Compare DynaFlow CA contact freq to ESMDynamic, aligned by PDB resseq.

  Aligning by residue number (not index-wise) matters because the Ago2 PDBs start
  at resseq 22 (4W5N) / 410 (9K6T) — an index-wise slice would compare residue 22
  of DynaFlow against residue 1 of ESMDynamic, destroying the correlation.
  """
  n_esm = esm_freq.shape[0]
  valid, i = esm_indices(dyn["resnums"], n_esm)
  f_df = dyn["freq"][np.ix_(valid, valid)]
  e_df = esm_freq[np.ix_(i, i)]
  L = len(i)
  if L < 4:
    return dict(freq_pearson=0.0, L=L)
  tri = np.triu_indices(L, k=3)
  v_d, v_e = f_df[tri], e_df[tri]
  pearson = float(np.corrcoef(v_d, v_e)[0, 1]) if v_d.std() > 0 and v_e.std() > 0 else 0.0
  out = dict(freq_pearson=round(pearson, 4), L=L)
  if esm_dyn is not None:
    e_dyn = esm_dyn[np.ix_(i, i)]
    d_dyn = dyn["dynamic_mask"][np.ix_(valid, valid)]
    # overlap of dynamic pairs (upper triangle)
    active = (e_dyn[tri] > 0.5) | d_dyn[tri]
    if active.sum() > 0:
      agree = ((e_dyn[tri] > 0.5) == d_dyn[tri])[active].mean()
      out["dynamic_agreement"] = round(float(agree), 4)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--samples-dir", required=True)
  ap.add_argument("--esm-dir", default="", help="ESMDynamic output_dir")
  ap.add_argument("--esm-id", default="AGO2", help="fallback protein ID in ESMDynamic outputs")
  ap.add_argument("--esm-id-map", default="",
                  help="per-system esm-id, '4W5N:hAgo2_chainA,9K6T:hAgo2_9K6T_chainA_res410-859'")
  ap.add_argument("--out", default="")
  ap.add_argument("--dynaflow-only", action="store_true")
  ap.add_argument("--systems", default="4W5N,9K6T")
  args = ap.parse_args()

  out_dir = args.out or os.path.join(RESULTS, "ago2", "esmdynamic")
  os.makedirs(out_dir, exist_ok=True)

  rows = []
  for pid in [s.strip().upper() for s in args.systems.split(",") if s.strip()]:
    npz = os.path.join(args.samples_dir, f"{pid}.npz")
    pdb = _find_pdb(pid)
    if not os.path.exists(npz) or not pdb:
      print(f"[skip] {pid}: missing npz or pdb")
      continue
    dyn = analyze_samples(npz, pdb)
    plot_contact_map(dyn["freq"], f"{pid} DynaFlow contact freq",
                     os.path.join(out_dir, f"{pid}_contact_freq.png"))
    plot_contact_map(dyn["dynamic_mask"].astype(float),
                     f"{pid} DynaFlow dynamic pairs",
                     os.path.join(out_dir, f"{pid}_dynamic.png"), vmax=1.0)
    row = dict(pdb_id=pid, n_ca=dyn["n_ca"], n_dynamic_pairs=dyn["n_dynamic_pairs"],
               mean_contact_freq=round(dyn["mean_freq"], 4))
    if args.esm_dir and not args.dynaflow_only:
      esm_id = args.esm_id
      if args.esm_id_map:
        for pair in args.esm_id_map.split(","):
          k, _, v = pair.partition(":")
          if k.strip().upper() == pid:
            esm_id = v.strip()
      esm_f = load_esmdynamic_frequency(args.esm_dir, esm_id)
      esm_d = load_esmdynamic_dynamic_prob(args.esm_dir, esm_id)
      if esm_f is not None:
        cmp = compare_to_esm(dyn, esm_f, esm_d)
        row.update(cmp)
        # plot the residue-aligned submatrix (ESM index = PDB residue order)
        valid, i = esm_indices(dyn["resnums"], esm_f.shape[0])
        if len(i) >= 4:
          plot_contact_map(esm_f[np.ix_(i, i)],
                           f"ESMDynamic freq ({esm_id})",
                           os.path.join(out_dir, f"{pid}_esm_freq.png"))
      else:
        print(f"[warn] no ESMDynamic frequency for {esm_id} in {args.esm_dir}")
    rows.append(row)

  if len(rows) >= 2:
    # diff map between conditions
    pids = [r["pdb_id"] for r in rows]
    if len(pids) == 2:
      d0 = analyze_samples(os.path.join(args.samples_dir, f"{pids[0]}.npz"), _find_pdb(pids[0]))
      d1 = analyze_samples(os.path.join(args.samples_dir, f"{pids[1]}.npz"), _find_pdb(pids[1]))
      L = min(d0["freq"].shape[0], d1["freq"].shape[0])
      diff = np.abs(d0["freq"][:L, :L] - d1["freq"][:L, :L])
      plot_contact_map(diff, f"Contact freq |{pids[0]} - {pids[1]}|",
                       os.path.join(out_dir, "contact_freq_diff.png"), vmax=0.5)

  df = pd.DataFrame(rows)
  df.to_csv(os.path.join(out_dir, "overlay_summary.csv"), index=False)
  print(df.to_string(index=False))
  print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
  main()
