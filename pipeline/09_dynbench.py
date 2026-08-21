#!/usr/bin/env python3
"""09_dynbench.py — dynamic ensemble benchmark for protein-RNA flow matching.

Evaluates generated conformational ensembles against MD reference (and NMR when
available) across four tiers:
  A: amplitude fidelity (RMSF, RMSD Wasserstein)
  B: collective motion (DCCM, RMSIP, contact frequency)
  C: phase-space coverage (PCA macrostate populations, KDE overlap, BioEmu-style
     MD-frame coverage curve, 2D free-energy landscape MAE in kcal/mol)
  D: physics validity (energy distribution, clash rate)

Includes a null-Gaussian baseline (per-atom independent noise with MD-matched RMSF)
that Tier A cannot distinguish from a real model but Tier B/C can.

Inputs:
  - Generated samples: results/samples/<ckpt>/<PDB>.npz  (gen_coords: n_gen,A,3)
  - MD reference: data/md/<PDB>.npz (coords, static, atom_elements, atom_residx, res_restype)
  - NMR (optional): data/nmr/<PDB>.npz

All coordinates are pre-aligned (gmx trjconv -pbc cluster + -fit rot+trans).
No PBC correction or Kabsch alignment needed here.

Run:  python3 09_dynbench.py --samples results/samples/flow_model_r3
      python3 09_dynbench.py --samples results/samples/flow_model_r3 --n-null 200
"""
import argparse, os, sys
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform, pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))


# ============================================================
# Tier A: Amplitude fidelity
# ============================================================

def rmsf(coords):
    """Per-atom RMSF from (F,A,3) coords -> (A,)."""
    mu = coords.mean(axis=0, keepdims=True)
    return np.sqrt(((coords - mu) ** 2).sum(-1).mean(0))


def rmsd_to_ref(coords, ref):
    """Per-frame RMSD to a reference structure. coords (F,A,3), ref (A,3) -> (F,)."""
    return np.sqrt(((coords - ref[None]) ** 2).sum(-1).mean(-1))


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def spearman(a, b):
    r, _ = stats.spearmanr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def wasserstein_1d(a, b, n=200):
    qs = np.linspace(0, 1, n)
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def block_ci(coords, n_blocks=5):
    """95% CI of mean RMSF from block averaging. Returns (mean, lo, hi)."""
    F = len(coords)
    bs = F // n_blocks
    block_rmsfs = [rmsf(coords[i*bs:(i+1)*bs]).mean() for i in range(n_blocks)]
    m = np.mean(block_rmsfs)
    se = np.std(block_rmsfs, ddof=1) / np.sqrt(n_blocks)
    return float(m), float(m - 1.96*se), float(m + 1.96*se)


def tier_a(md_coords, gen_coords, static):
    """Compute Tier A metrics. All inputs in absolute coords (Angstrom)."""
    r_md = rmsf(md_coords)
    r_gen = rmsf(gen_coords)
    d_md = rmsd_to_ref(md_coords, static)
    d_gen = rmsd_to_ref(gen_coords, static)
    md_mean, md_lo, md_hi = block_ci(md_coords)
    gen_mean = float(r_gen.mean())
    ratio = gen_mean / md_mean if md_mean > 0 else 0.0
    ratio_in_ci = md_lo <= gen_mean <= md_hi
    return dict(
        rmsf_pearson=round(pearson(r_md, r_gen), 4),
        rmsf_spearman=round(spearman(r_md, r_gen), 4),
        mean_rmsf_md=round(md_mean, 3),
        mean_rmsf_gen=round(gen_mean, 3),
        rmsf_ratio=round(ratio, 3),
        rmsf_ratio_in_ci=ratio_in_ci,
        rmsd_wass=round(wasserstein_1d(d_md, d_gen), 3),
    )


def generate_null_gaussian(static, md_coords, n_samples):
    """Null baseline: per-atom independent Gaussian with MD-matched RMSF."""
    r_md = rmsf(md_coords)
    noise = np.random.randn(n_samples, len(static), 3)
    sigma_per_atom = r_md / np.sqrt(3.0)
    return static[None] + noise * sigma_per_atom[None, :, None]


# ============================================================
# Tier B: Collective motion
# ============================================================

def dccm(coords):
    """Dynamic cross-correlation matrix. coords (F,A,3) -> (A,A) in [-1,1]."""
    F, A, _ = coords.shape
    mu = coords.mean(0, keepdims=True)
    delta = (coords - mu).reshape(F, A, 3)  # (F,A,3)
    # C_ij = <dr_i . dr_j> / (sigma_i * sigma_j)
    # dr_i . dr_j summed over xyz -> (F,A,A)
    dot = np.einsum('fai,fbi->ab', delta, delta) / F
    sigma = np.sqrt((delta ** 2).sum(-1).mean(0))  # (A,)
    denom = np.outer(sigma, sigma)
    denom = np.where(denom > 1e-12, denom, 1e-12)
    return dot / denom


def dccm_compare(C_md, C_gen):
    """Compare two DCCM matrices: upper-triangle Pearson + Frobenius error."""
    idx = np.triu_indices(len(C_md), k=1)
    v_md = C_md[idx]
    v_gen = C_gen[idx]
    p = pearson(v_md, v_gen)
    frob = float(np.sqrt(((C_gen - C_md) ** 2).sum()))
    frob_norm = frob / (np.sqrt((C_md ** 2).sum()) + 1e-12)
    return p, frob_norm


def residue_dccm(coords, atom_residx, res_restype, atom_names=None):
    """Residue-level DCCM. Uses CA (protein) / C4' (RNA) when atom_names available,
    otherwise falls back to residue center-of-mass.
    Returns (C_res, n_residues) or (None, None) if not enough residues."""
    n_res = int(atom_residx.max()) + 1
    if n_res < 5:
        return None, None
    F = coords.shape[0]

    if atom_names is not None:
        # Select representative atoms: CA for protein, C4' for RNA
        rep_atoms = []
        for r in range(n_res):
            mask = np.where(atom_residx == r)[0]
            if len(mask) == 0:
                continue
            names_r = atom_names[mask]
            is_protein = res_restype[r] < 20
            target = "CA" if is_protein else "C4'"
            hit = np.where(names_r == target)[0]
            if len(hit) > 0:
                rep_atoms.append(mask[hit[0]])
            else:
                rep_atoms.append(mask[0])
        rep_atoms = np.array(rep_atoms)
        sub_coords = coords[:, rep_atoms]
        return dccm(sub_coords), n_res

    # Fallback: residue center-of-mass
    res_coords = np.zeros((F, n_res, 3), dtype=np.float32)
    for r in range(n_res):
        mask = np.where(atom_residx == r)[0]
        if len(mask) == 0:
            continue
        res_coords[:, r] = coords[:, mask].mean(axis=1)
    return dccm(res_coords), n_res


def rmsip(coords_a, coords_b, k=10):
    """Root Mean Square Inner Product of top-k PC subspaces.
    coords_a/b: (F,A,3). Returns float in [0,1]."""
    def top_k_pcs(c, k):
        X = c.reshape(len(c), -1)
        X = X - X.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt[:k]  # (k, 3A)
    Va = top_k_pcs(coords_a, k)
    Vb = top_k_pcs(coords_b, k)
    overlap = (Va @ Vb.T) ** 2  # (k,k)
    return float(np.sqrt(overlap.sum() / k))


def contact_frequency(coords, static, cutoff=6.0, max_static_dist=12.0):
    """Contact frequencies for non-bonded heavy atom pairs.
    Returns (freq array, pair_indices (N,2), rna_prot_mask (N,) bool)."""
    A = static.shape[0]
    # Find candidate pairs: within max_static_dist in static structure
    sd = squareform(pdist(static))
    # Exclude self and immediate bonded neighbors (sequential atoms, dist < 2A)
    mask = (sd < max_static_dist) & (sd > 2.0)
    pairs = np.array(np.where(np.triu(mask, k=1))).T  # (N,2)
    if len(pairs) == 0:
        return np.array([]), pairs, np.array([], dtype=bool)
    # Compute per-frame distances for these pairs
    F = coords.shape[0]
    freq = np.zeros(len(pairs), dtype=np.float32)
    chunk = 50
    for start in range(0, F, chunk):
        end = min(start + chunk, F)
        c = coords[start:end]  # (chunk, A, 3)
        d = np.sqrt(((c[:, pairs[:, 0]] - c[:, pairs[:, 1]]) ** 2).sum(-1))  # (chunk, N)
        freq += (d < cutoff).sum(0)
    freq /= F
    return freq, pairs, None  # rna_prot_mask filled by caller


def identify_rna_protein_contacts(pairs, atom_residx, res_restype):
    """Identify which contact pairs are RNA-protein interface contacts.
    Protein residues: restype 0-19, RNA: 20-23."""
    res_a = res_restype[atom_residx[pairs[:, 0]]]
    res_b = res_restype[atom_residx[pairs[:, 1]]]
    is_prot_a = res_a < 20
    is_rna_a = (res_a >= 20) & (res_a < 24)
    is_prot_b = res_b < 20
    is_rna_b = (res_b >= 20) & (res_b < 24)
    return (is_prot_a & is_rna_b) | (is_rna_a & is_prot_b)


def tier_b(md_coords, gen_coords, static, atom_residx, res_restype, atom_names=None):
    """Compute Tier B metrics."""
    results = {}
    # DCCM atom-level
    C_md = dccm(md_coords)
    C_gen = dccm(gen_coords)
    dp, df = dccm_compare(C_md, C_gen)
    results["dccm_pearson"] = round(dp, 4)
    results["dccm_frob_err"] = round(df, 4)

    # DCCM residue-level
    C_md_res, _ = residue_dccm(md_coords, atom_residx, res_restype, atom_names)
    C_gen_res, _ = residue_dccm(gen_coords, atom_residx, res_restype, atom_names)
    if C_md_res is not None and C_gen_res is not None:
        dp_r, df_r = dccm_compare(C_md_res, C_gen_res)
        results["dccm_res_pearson"] = round(dp_r, 4)
        results["dccm_res_frob_err"] = round(df_r, 4)

    # RMSIP
    results["rmsip_10"] = round(rmsip(md_coords, gen_coords, k=10), 4)

    # Contact frequency
    freq_md, pairs, _ = contact_frequency(md_coords, static)
    if len(pairs) > 0:
        freq_gen, _, _ = contact_frequency(gen_coords, static)
        active = (freq_md > 0.01) | (freq_gen > 0.01)
        if active.sum() > 10:
            results["contact_pearson"] = round(pearson(freq_md[active], freq_gen[active]), 4)
            # RNA-protein subset
            rp_mask = identify_rna_protein_contacts(pairs, atom_residx, res_restype)
            rp_active = active & rp_mask
            if rp_active.sum() > 5:
                results["contact_rna_prot_pearson"] = round(
                    pearson(freq_md[rp_active], freq_gen[rp_active]), 4)
    return results, C_md, C_gen


# ============================================================
# Tier C: Phase-space coverage
# ============================================================

def pca_project(md_coords, gen_coords, k=5):
    """Project gen into MD-defined PC space. Returns (md_pc, gen_pc, explained_var_ratio)."""
    X_md = md_coords.reshape(len(md_coords), -1)
    mu = X_md.mean(0)
    X_md_c = X_md - mu
    _, S, Vt = np.linalg.svd(X_md_c, full_matrices=False)
    var_explained = (S[:k] ** 2) / (S ** 2).sum()
    P = Vt[:k].T  # (3A, k)
    md_pc = X_md_c @ P
    gen_pc = (gen_coords.reshape(len(gen_coords), -1) - mu) @ P
    return md_pc, gen_pc, var_explained


def macrostate_js(md_pc, gen_pc, n_clusters=5):
    """K-means clustering in PC space, compare population vectors via JS divergence."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(md_pc)
    labels_md = km.predict(md_pc)
    labels_gen = km.predict(gen_pc)
    pop_md = np.bincount(labels_md, minlength=n_clusters).astype(float)
    pop_gen = np.bincount(labels_gen, minlength=n_clusters).astype(float)
    pop_md /= pop_md.sum()
    pop_gen /= pop_gen.sum() + 1e-12
    # JS divergence
    m = 0.5 * (pop_md + pop_gen)
    def kl(p, q):
        mask = p > 0
        return float((p[mask] * np.log(p[mask] / (q[mask] + 1e-12))).sum())
    js = 0.5 * kl(pop_md, m) + 0.5 * kl(pop_gen, m)
    return js, pop_md, pop_gen


def kde_bhattacharyya(md_pc2, gen_pc2, grid_size=80):
    """Bhattacharyya coefficient of KDE densities in PC1-PC2 space."""
    from scipy.stats import gaussian_kde
    all_pts = np.vstack([md_pc2, gen_pc2])
    xmin, ymin = all_pts.min(0) - 1.0
    xmax, ymax = all_pts.max(0) + 1.0
    xx, yy = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    try:
        kde_md = gaussian_kde(md_pc2.T)(positions)
        kde_gen = gaussian_kde(gen_pc2.T)(positions)
    except np.linalg.LinAlgError:
        return 0.0
    kde_md /= kde_md.sum() + 1e-12
    kde_gen /= kde_gen.sum() + 1e-12
    bc = float(np.sqrt(kde_md * kde_gen).sum())
    return bc


# RMSD thresholds (A) for the coverage curve; scalars reported at 3/5 A + AUC.
COV_THRESHOLDS = np.arange(0.5, 8.01, 0.25)
KT_KCAL = 0.596  # k_B T at 300 K in kcal/mol


def _cov_auc(cov, thresholds=COV_THRESHOLDS):
    """Normalized trapezoidal AUC of a coverage curve, in [0,1]."""
    return float((((cov[:-1] + cov[1:]) / 2) * np.diff(thresholds)).sum()
                 / (thresholds[-1] - thresholds[0]))


def coverage_curve(md_coords, gen_coords, thresholds=COV_THRESHOLDS, min_frac=0.001):
    """BioEmu-style coverage: fraction of MD reference frames covered by the
    generated ensemble. A frame is covered when at least min_frac (0.1%) of the
    samples lie within the RMSD threshold. Aligned-frame all-heavy-atom RMSD
    (no per-pair refit, consistent with every other metric here).
    Returns coverage array of len(thresholds)."""
    n_gen = len(gen_coords)
    k = max(1, int(np.ceil(min_frac * n_gen)))  # need >= k samples within threshold
    dk = np.empty(len(md_coords), dtype=np.float32)
    for f in range(len(md_coords)):
        d = np.sqrt(((gen_coords - md_coords[f][None]) ** 2).sum(-1).mean(-1))
        dk[f] = np.partition(d, k - 1)[k - 1]  # k-th smallest sample RMSD
    return np.array([float((dk < t).mean()) for t in thresholds])


def free_energy_mae(md_pc2, gen_pc2, grid_size=40):
    """2D free-energy landscape MAE on MD-defined PC1-PC2 (BioEmu-style).
    F = -ln(P/P_max) in k_BT; empty bins get a half-count pseudocount so missing
    regions are penalized at a finite cap. MAE over MD-occupied bins.
    Returns (mae in kcal/mol, F_md, F_gen, (xedges, yedges)); F in k_BT."""
    all_pts = np.vstack([md_pc2, gen_pc2])
    lo, hi = all_pts.min(0) - 1e-6, all_pts.max(0) + 1e-6
    rng = [[lo[0], hi[0]], [lo[1], hi[1]]]
    H_md, xe, ye = np.histogram2d(md_pc2[:, 0], md_pc2[:, 1], bins=grid_size, range=rng)
    H_gen, _, _ = np.histogram2d(gen_pc2[:, 0], gen_pc2[:, 1], bins=grid_size, range=rng)

    def fe(H):
        P = H / H.sum()
        P = np.where(P > 0, P, 0.5 / H.sum())
        return -np.log(P / P.max())

    F_md, F_gen = fe(H_md), fe(H_gen)
    occ = H_md > 0
    mae = float(np.abs(F_md[occ] - F_gen[occ]).mean()) * KT_KCAL
    return mae, F_md, F_gen, (xe, ye), occ, (H_gen > 0)


def tier_c(md_coords, gen_coords):
    """Compute Tier C metrics. Returns (results, md_pc, gen_pc, extras)."""
    results = {}
    md_pc, gen_pc, var_ratio = pca_project(md_coords, gen_coords, k=5)
    results["pca_var_top5"] = round(float(var_ratio.sum()), 4)

    # Macrostate JS
    try:
        js, pop_md, pop_gen = macrostate_js(md_pc, gen_pc, n_clusters=5)
        results["macrostate_js"] = round(js, 4)
        pop_ratio = pop_gen / (pop_md + 1e-12)
        results["macrostate_in_tolerance"] = int(((pop_ratio > 0.3) & (pop_ratio < 3.0)).sum())
    except Exception:
        pass

    # KDE Bhattacharyya in PC1-PC2
    bc = kde_bhattacharyya(md_pc[:, :2], gen_pc[:, :2])
    results["kde_bc_pc12"] = round(bc, 4)

    # BioEmu-style MD-frame coverage (0.1% sample rule)
    cov = coverage_curve(md_coords, gen_coords)
    i3 = int(np.where(np.isclose(COV_THRESHOLDS, 3.0))[0][0])
    i5 = int(np.where(np.isclose(COV_THRESHOLDS, 5.0))[0][0])
    results["cov_3A"] = round(float(cov[i3]), 4)
    results["cov_5A"] = round(float(cov[i5]), 4)
    results["cov_auc"] = round(_cov_auc(cov), 4)

    # 2D free-energy landscape MAE (kcal/mol)
    mae, F_md, F_gen, edges, occ_md, occ_gen = free_energy_mae(md_pc[:, :2], gen_pc[:, :2])
    results["fe_mae_kcal"] = round(mae, 3)

    extras = dict(cov_curve=cov, cov_thresholds=COV_THRESHOLDS,
                  F_md=F_md, F_gen=F_gen, fe_edges=edges,
                  fe_occ_md=occ_md, fe_occ_gen=occ_gen)
    return results, md_pc, gen_pc, extras


# ============================================================
# Tier D: Physics validity (lightweight)
# ============================================================

def clash_rate(coords, atom_residx, static=None, threshold=1.2, sample_frames=50):
    """Fraction of frames with steric clashes (non-bonded heavy atom pairs < threshold).
    Excludes intra-residue pairs (which include bonded 1-2/1-3 neighbors at ~1.2-1.5A).
    Only checks pairs within 4A in the static structure for efficiency."""
    F, A, _ = coords.shape
    idx = np.random.choice(F, size=min(sample_frames, F), replace=False)
    # Find candidate inter-residue pairs close in static structure
    if static is not None:
        sd = squareform(pdist(static))
    else:
        sd = squareform(pdist(coords[0]))
    ri, rj = np.where(np.triu(sd < 4.0, k=1))
    inter = atom_residx[ri] != atom_residx[rj]
    pairs = np.column_stack([ri[inter], rj[inter]])
    if len(pairs) == 0:
        return 0.0
    n_clash_frames = 0
    for i in idx:
        dists = np.sqrt(((coords[i, pairs[:, 0]] - coords[i, pairs[:, 1]]) ** 2).sum(-1))
        if (dists < threshold).any():
            n_clash_frames += 1
    return n_clash_frames / len(idx)


def tier_d(gen_coords, atom_residx, static=None):
    """Compute Tier D metrics (no energy head version)."""
    results = {}
    results["clash_rate"] = round(clash_rate(gen_coords, atom_residx, static), 4)
    return results


# ============================================================
# Visualization
# ============================================================

def plot_dccm_comparison(C_md, C_gen, pid, figdir):
    """Side-by-side DCCM heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    vmax = max(abs(C_md).max(), abs(C_gen).max(), 0.3)
    ax1.imshow(C_md, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax1.set_title(f"{pid} DCCM (MD)")
    ax2.imshow(C_gen, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax2.set_title(f"{pid} DCCM (Gen)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{pid}_dccm.png"), dpi=300)
    plt.close(fig)


def plot_pca_scatter(md_pc, gen_pc, pid, figdir):
    """PCA scatter (PC1 vs PC2)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(md_pc[:, 0], md_pc[:, 1], s=6, alpha=0.4, label="MD")
    ax.scatter(gen_pc[:, 0], gen_pc[:, 1], s=6, alpha=0.4, label="Gen")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8); ax.set_title(f"{pid} PC1-PC2")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{pid}_pca.png"), dpi=300)
    plt.close(fig)


def plot_rmsf_profile(r_md, r_gen, r_null, pid, figdir, r_nmr=None):
    """RMSF per-atom profile comparison."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(r_md))
    ax.plot(x, r_md, lw=1.4, label="MD")
    ax.plot(x, r_gen, lw=1.4, alpha=0.8, label="Gen")
    ax.plot(x, r_null, lw=1.0, alpha=0.6, ls=":", label="Null")
    if r_nmr is not None:
        ax.plot(x, r_nmr, lw=1.2, ls="--", label="NMR")
    ax.set_xlabel("atom index"); ax.set_ylabel("RMSF (A)")
    ax.legend(fontsize=8); ax.set_title(f"{pid} atom RMSF")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{pid}_rmsf.png"), dpi=300)
    plt.close(fig)


def plot_coverage_curve(cov_gen, cov_null, cov_self, thresholds, pid, figdir):
    """BioEmu-style coverage vs RMSD threshold (0.1% sample rule).
    MD-self (half vs half) is the empirical ceiling, null the floor."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thresholds, cov_gen, "o-", ms=3, label="Gen")
    if cov_null is not None:
        ax.plot(thresholds, cov_null, "s--", ms=3, alpha=0.7, label="Null")
    if cov_self is not None:
        ax.plot(thresholds, cov_self, "^-", ms=3, alpha=0.8, color="gray", label="MD self")
    ax.set_xlabel("RMSD threshold (A)"); ax.set_ylabel("MD frame coverage")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8); ax.set_title(f"{pid} coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{pid}_coverage.png"), dpi=300)
    plt.close(fig)


def plot_free_energy(F_md, F_gen, edges, occ_md, occ_gen, pid, figdir):
    """2D free-energy landscapes (k_BT) on PC1-PC2, MD vs Gen.
    Unoccupied bins are masked white (the MAE metric still penalizes them)."""
    xe, ye = edges
    vmax = float(min(6.0, max(np.percentile(F_md[occ_md], 99),
                              np.percentile(F_gen[occ_gen], 99), 1.0)))
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad("white")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True,
                                 constrained_layout=True)
    im = None
    for ax, F, occ, tag in [(a1, F_md, occ_md, "MD"), (a2, F_gen, occ_gen, "Gen")]:
        Fm = np.ma.masked_where(~occ, F)
        im = ax.pcolormesh(xe, ye, Fm.T, cmap=cmap, vmin=0, vmax=vmax, shading="auto")
        ax.set_xlabel("PC1 (A)"); ax.set_title(f"{pid} {tag}")
        ax.set_facecolor("white")
    a1.set_ylabel("PC2 (A)")
    fig.colorbar(im, ax=[a1, a2], shrink=0.85, label="F (k$_B$T)")
    fig.savefig(os.path.join(figdir, f"{pid}_free_energy.png"), dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def load_md(pid):
    """Load MD reference npz. Returns dict or None."""
    p = os.path.join(DATA, "md", f"{pid}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = dict(coords=z["coords"].astype(np.float32),
             static=z["static"].astype(np.float32),
             atom_elements=z["atom_elements"],
             atom_residx=z["atom_residx"].astype(np.int64),
             res_restype=z["res_restype"].astype(np.int64),
             pdb_id=str(z["pdb_id"]))
    if "atom_names" in z:
        d["atom_names"] = z["atom_names"]
    return d


def load_gen(samples_dir, pid):
    """Load generated samples npz."""
    p = os.path.join(samples_dir, f"{pid}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["gen_coords"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="DynaFlow dynamic ensemble benchmark")
    ap.add_argument("--samples", required=True,
                    help="directory with generated sample npz files (per system)")
    ap.add_argument("--systems", default="",
                    help="comma-sep PDB IDs (else auto-detect from samples dir)")
    ap.add_argument("--n-null", type=int, default=200,
                    help="number of null-Gaussian samples to generate")
    ap.add_argument("--out", default="",
                    help="output directory (else results/dynbench/<samples_stem>)")
    args = ap.parse_args()

    samples_dir = args.samples
    if not os.path.isdir(samples_dir):
        print(f"ERROR: samples directory not found: {samples_dir}"); sys.exit(1)

    # Determine systems
    if args.systems:
        pids = [s.strip().upper() for s in args.systems.split(",") if s.strip()]
    else:
        pids = sorted([f.replace(".npz", "").upper()
                       for f in os.listdir(samples_dir) if f.endswith(".npz")])
    if not pids:
        print("ERROR: no systems found"); sys.exit(1)

    # Output directory
    stem = os.path.basename(samples_dir.rstrip("/"))
    outdir = args.out if args.out else os.path.join(RESULTS, "dynbench", stem)
    figdir = os.path.join(outdir, "figs")
    os.makedirs(figdir, exist_ok=True)

    print(f"dynbench | samples={samples_dir} | systems={pids} | n_null={args.n_null}")
    rows = []
    null_rows = []

    for pid in pids:
        md = load_md(pid)
        if md is None:
            print(f"[skip] {pid}: no MD npz"); continue
        gen = load_gen(samples_dir, pid)
        if gen is None:
            print(f"[skip] {pid}: no generated samples"); continue

        md_coords = md["coords"]
        static = md["static"]
        n_gen = len(gen)

        if gen.shape[1] != md_coords.shape[1]:
            print(f"[skip] {pid}: atom mismatch gen={gen.shape[1]} md={md_coords.shape[1]}")
            continue

        if n_gen < 100:
            print(f"[warn] {pid}: n_gen={n_gen} < 100, DCCM/RMSIP may be unstable")

        print(f"\n{'='*60}\n{pid} | A={md_coords.shape[1]} | F_md={len(md_coords)} | n_gen={n_gen}")

        # Generate null baseline
        null_coords = generate_null_gaussian(static, md_coords, args.n_null)

        # --- Tier A ---
        row_a = tier_a(md_coords, gen, static)
        null_a = tier_a(md_coords, null_coords, static)

        # --- Tier B ---
        anames = md.get("atom_names")
        row_b, C_md, C_gen = tier_b(md_coords, gen, static,
                                     md["atom_residx"], md["res_restype"], anames)
        null_b, _, C_null = tier_b(md_coords, null_coords, static,
                                    md["atom_residx"], md["res_restype"], anames)

        # --- Tier C ---
        row_c, md_pc, gen_pc, xc = tier_c(md_coords, gen)
        null_c, _, null_pc, xn = tier_c(md_coords, null_coords)

        # MD self-reference (empirical ceiling): first half vs second half
        fh = len(md_coords) // 2
        cov_self = coverage_curve(md_coords[:fh], md_coords[fh:])
        fe_self, _, _, _, _, _ = free_energy_mae(md_pc[:fh, :2], md_pc[fh:, :2])
        row_c["cov_self_5A"] = round(float(cov_self[int(np.where(np.isclose(COV_THRESHOLDS, 5.0))[0][0])]), 4)
        row_c["cov_self_auc"] = round(_cov_auc(cov_self), 4)
        row_c["fe_self_kcal"] = round(fe_self, 3)

        # --- Tier D ---
        row_d = tier_d(gen, md["atom_residx"], static)

        # Combine
        row = dict(pdb_id=pid, n_atoms=int(md_coords.shape[1]), n_gen=n_gen)
        row.update(row_a); row.update(row_b); row.update(row_c); row.update(row_d)
        rows.append(row)

        null_row = dict(pdb_id=pid, n_atoms=int(md_coords.shape[1]), n_gen=args.n_null)
        null_row.update(null_a); null_row.update(null_b); null_row.update(null_c)
        null_row.update(tier_d(null_coords, md["atom_residx"], static))
        null_rows.append(null_row)

        # Print comparison
        print(f"  Tier A: rmsf_pears={row_a['rmsf_pearson']:.3f} (null={null_a['rmsf_pearson']:.3f})"
              f"  ratio={row_a['rmsf_ratio']:.2f}  wass={row_a['rmsd_wass']:.3f}")
        print(f"  Tier B: dccm={row_b.get('dccm_pearson','?')}"
              f" (null={null_b.get('dccm_pearson','?')})"
              f"  rmsip={row_b.get('rmsip_10','?')}"
              f"  contact={row_b.get('contact_pearson','?')}")
        print(f"  Tier C: macro_js={row_c.get('macrostate_js','?')}"
              f"  kde_bc={row_c.get('kde_bc_pc12','?')}"
              f"  cov5A={row_c.get('cov_5A','?')} (null={null_c.get('cov_5A','?')}"
              f" self={row_c.get('cov_self_5A','?')})"
              f"  fe_mae={row_c.get('fe_mae_kcal','?')} kcal/mol"
              f" (null={null_c.get('fe_mae_kcal','?')} self={row_c.get('fe_self_kcal','?')})")
        print(f"  Tier D: clash={row_d['clash_rate']:.3f}")

        # --- NMR bonus ---
        nmr_p = os.path.join(DATA, "nmr", f"{pid}.npz")
        r_nmr = None
        if os.path.exists(nmr_p):
            nmr_coords = np.load(nmr_p)["coords"].astype(np.float32)
            if nmr_coords.shape[1] == md_coords.shape[1]:
                r_nmr = rmsf(nmr_coords)
                row["rmsf_pearson_nmr"] = round(pearson(r_nmr, rmsf(gen)), 4)
                print(f"  NMR: rmsf_pears_nmr={row['rmsf_pearson_nmr']:.3f}")
            else:
                print(f"  [nmr-skip] atoms {nmr_coords.shape[1]} != {md_coords.shape[1]}")

        # --- Figures ---
        # Downsample DCCM for large systems (>500 atoms -> show residue-level)
        if C_md.shape[0] <= 500:
            plot_dccm_comparison(C_md, C_gen, pid, figdir)
        else:
            C_md_r, _ = residue_dccm(md_coords, md["atom_residx"], md["res_restype"], anames)
            C_gen_r, _ = residue_dccm(gen, md["atom_residx"], md["res_restype"], anames)
            if C_md_r is not None:
                plot_dccm_comparison(C_md_r, C_gen_r, pid, figdir)
        plot_pca_scatter(md_pc, gen_pc, pid, figdir)
        plot_rmsf_profile(rmsf(md_coords), rmsf(gen), rmsf(null_coords), pid, figdir, r_nmr)
        plot_coverage_curve(xc["cov_curve"], xn["cov_curve"], cov_self,
                            xc["cov_thresholds"], pid, figdir)
        plot_free_energy(xc["F_md"], xc["F_gen"], xc["fe_edges"],
                         xc["fe_occ_md"], xc["fe_occ_gen"], pid, figdir)

    # --- Output CSVs ---
    if not rows:
        print("\nNo systems evaluated."); return

    import pandas as pd
    df = pd.DataFrame(rows)
    df_null = pd.DataFrame(null_rows)

    df.to_csv(os.path.join(outdir, "per_system.csv"), index=False)
    df_null.to_csv(os.path.join(outdir, "per_system_null.csv"), index=False)

    # Summary by group
    # Determine NMR vs X-ray from manifest
    manifest_p = os.path.join(DATA, "systems_manifest.csv")
    if os.path.exists(manifest_p):
        man = pd.read_csv(manifest_p)
        nmr_ids = set(man[man.method == "NMR"].pdb_id.str.upper())
    else:
        nmr_ids = set()

    df["group"] = df.pdb_id.apply(lambda x: "NMR" if x in nmr_ids else "X-ray")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    summary_rows = []
    for grp in ["NMR", "X-ray", "ALL"]:
        sub = df if grp == "ALL" else df[df.group == grp]
        if len(sub) == 0:
            continue
        r = {"group": grp, "n_systems": len(sub)}
        for c in numeric_cols:
            if c in ("n_atoms", "n_gen"):
                continue
            r[c] = round(float(sub[c].mean()), 4)
        summary_rows.append(r)

    # Also add null baseline summary
    for grp_label in ["NULL"]:
        r = {"group": grp_label, "n_systems": len(df_null)}
        for c in numeric_cols:
            if c in ("n_atoms", "n_gen") or c not in df_null.columns:
                continue
            r[c] = round(float(df_null[c].mean()), 4)
        summary_rows.append(r)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    print(f"\n{'='*60}")
    print("SUMMARY (model vs null-Gaussian baseline):")
    print(df_summary.to_string(index=False))
    print(f"\nResults saved to: {outdir}")


if __name__ == "__main__":
    main()
