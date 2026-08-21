#!/usr/bin/env python3
"""
00_select_systems.py — data cleaning + MD system selection + train/test split.

Scans every complex in "RNA-protein complexes/<PDBID>/<pdbid>.pdb":
  * experiment method (X-ray / NMR), number of NMR models
  * RNA length (nt), protein length (aa), solute atom count
  * interface size (RNA residues within 5 A of protein, model 1)
  * for NMR: RNA backbone fluctuation after protein-CA alignment (panel-c metric)
Selects MD candidates: small enough for the 3-day budget, NMR-dynamic first.
Writes data/systems_manifest.csv  (columns include selected / split / reason)

Stdlib + numpy only. Usage:
    python3 00_select_systems.py [--n-candidates 20] [--n-train 3] \
        [--max-rna-nt 60] [--max-prot-aa 400] [--max-solute-atoms 15000]
"""
import argparse, csv, os, re, sys
import numpy as np

RNA_RES = {"A", "U", "G", "C"}
AA_RES = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","HIE","HID",
          "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}

def parse_pdb(path):
    """Return dict with per-model atom records. Only standard ATOM records."""
    models, cur, method = [], [], "?"
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("EXPDTA"):
                method = line[10:].strip()
            elif line.startswith("MODEL"):
                cur = []
            elif line.startswith("ENDMDL"):
                if cur: models.append(cur); cur = []
            elif line.startswith("ATOM"):
                cur.append((line[12:16].strip(),      # atom name
                            line[17:20].strip(),      # resname
                            line[21],                 # chain
                            line[22:26].strip(),      # resseq (+icode via 26)
                            line[26].strip(),
                            float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if cur: models.append(cur)
    if not models: models = [[]]
    return method, models

def chain_residues(model):
    """Ordered unique residues: list of (chain, resseq+icode, resname)."""
    seen, order = set(), []
    for name, resn, ch, rs, ic, x, y, z in model:
        key = (ch, rs + ic)
        if key not in seen:
            seen.add(key); order.append((ch, rs + ic, resn))
    return order

def kabsch(mobile, ref):
    """Rotation+translation aligning mobile onto ref (Nx3)."""
    mc, rc = mobile.mean(0), ref.mean(0)
    H = (mobile - mc).T @ (ref - rc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, rc, mc

def nmr_fluctuation(models):
    """RNA backbone fluctuation (A) after aligning all models on protein CA.
    Returns (mean_fluct_A, n_rna_res) or (0, n) if <2 models."""
    def beads(model, want_rna):
        pts = {}
        for name, resn, ch, rs, ic, x, y, z in model:
            key = (ch, rs + ic)
            if resn in AA_RES and name == "CA" and not want_rna:
                pts[key] = (x, y, z)
            if resn in RNA_RES and want_rna and name in ("P", "C4'"):
                pts.setdefault(key, []).append((x, y, z))
        return pts
    ref_ca = beads(models[0], False)
    per_model_ca = [beads(m, False) for m in models]
    common_ca = [k for k in ref_ca if all(k in ca for ca in per_model_ca[1:])]
    if len(common_ca) < 5: return 0.0, 0
    ref_arr = np.array([ref_ca[k] for k in common_ca])
    # collect aligned RNA beads per model
    rna_keys = None; per_model = []
    for m, ca in zip(models, per_model_ca):
        mob = np.array([ca[k] for k in common_ca])
        R, rc, mc = kabsch(mob, ref_arr)
        rb = beads(m, True)
        keys = sorted(rb.keys())
        if rna_keys is None: rna_keys = keys
        if keys != rna_keys: return 0.0, len(rna_keys or [])
        # per-residue mean of P and C4'
        arr = np.array([np.mean(rb[k], axis=0) for k in keys])
        per_model.append((arr - mc) @ R.T + rc)
    if len(per_model) < 2 or not rna_keys: return 0.0, 0
    stack = np.stack(per_model)                      # (M, Nres, 3)
    fluct = np.sqrt(((stack - stack.mean(0))**2).sum(-1).mean(0))  # per-res
    return float(fluct.mean()), len(rna_keys)

def interface_size(model):
    """# RNA residues with any atom within 5 A of any protein atom (model 1)."""
    prot = np.array([(x, y, z) for n, r, c, s, i, x, y, z in model if r in AA_RES])
    rna_by_res = {}
    for n, r, c, s, i, x, y, z in model:
        if r in RNA_RES:
            rna_by_res.setdefault((c, s + i), []).append((x, y, z))
    if len(prot) == 0 or not rna_by_res: return 0
    count = 0
    for key, pts in rna_by_res.items():
        pts = np.array(pts)
        d2 = ((pts[:, None, :] - prot[None, :, :])**2).sum(-1)
        if (d2 < 25.0).any(): count += 1
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=3)
    ap.add_argument("--max-rna-nt", type=int, default=60)
    ap.add_argument("--max-prot-aa", type=int, default=400)
    ap.add_argument("--max-solute-atoms", type=int, default=15000)
    args = ap.parse_args()

    root = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdb_dir = os.environ.get("PDB_DIR", os.path.join(root, "RNA-protein complexes"))
    data_dir = os.environ.get("DATA_DIR", os.path.join(root, "data"))
    os.makedirs(data_dir, exist_ok=True)

    rows = []
    for entry in sorted(os.listdir(pdb_dir)):
        d = os.path.join(pdb_dir, entry)
        if not os.path.isdir(d): continue
        pdbs = [f for f in os.listdir(d)
                if f.lower().endswith(".pdb") and not f.startswith(".")]
        if not pdbs: continue
        pdb_id = os.path.splitext(pdbs[0])[0].upper()
        method, models = parse_pdb(os.path.join(d, pdbs[0]))
        m1 = models[0]
        residues = chain_residues(m1)
        rna_nt = sum(1 for _, _, r in residues if r in RNA_RES)
        prot_aa = sum(1 for _, _, r in residues if r in AA_RES)
        n_atoms = len(m1)
        is_nmr = "NMR" in method.upper()
        fluct, _ = nmr_fluctuation(models) if is_nmr and len(models) > 1 else (0.0, 0)
        iface = interface_size(m1)
        rows.append(dict(pdb_id=pdb_id, method=("NMR" if is_nmr else "X-ray"),
                         n_models=len(models), rna_nt=rna_nt, prot_aa=prot_aa,
                         solute_atoms=n_atoms, interface_res=iface,
                         nmr_fluct_A=round(fluct, 2)))

    # --- selection -------------------------------------------------------
    def qualifies(r):
        if r["rna_nt"] < 8 or r["rna_nt"] > args.max_rna_nt: return False
        if r["prot_aa"] < 20 or r["prot_aa"] > args.max_prot_aa: return False
        if r["solute_atoms"] > args.max_solute_atoms: return False
        if r["interface_res"] < 3: return False
        return True

    for r in rows:
        r["qualifies"] = qualifies(r)
        # priority: NMR with real motion first, then smaller systems
        r["score"] = (0 if (r["method"] == "NMR" and r["nmr_fluct_A"] >= 2.0) else 1,
                      r["solute_atoms"])

    cand = sorted([r for r in rows if r["qualifies"]], key=lambda r: r["score"])
    selected = cand[:args.n_candidates]
    sel_ids = {r["pdb_id"] for r in selected}
    # train = n_train smallest NMR-dynamic among selected; rest = test
    nmr_dyn = sorted([r for r in selected if r["method"] == "NMR" and r["nmr_fluct_A"] >= 2.0],
                     key=lambda r: r["solute_atoms"])
    train_ids = {r["pdb_id"] for r in nmr_dyn[:args.n_train]}
    if len(train_ids) < args.n_train:    # pad with smallest selected
        for r in sorted(selected, key=lambda r: r["solute_atoms"]):
            if len(train_ids) >= args.n_train: break
            train_ids.add(r["pdb_id"])

    for r in rows:
        r["selected"] = int(r["pdb_id"] in sel_ids)
        r["split"] = ("train" if r["pdb_id"] in train_ids
                      else "test" if r["pdb_id"] in sel_ids else "")

    out = os.path.join(data_dir, "systems_manifest.csv")
    cols = ["pdb_id","method","n_models","rna_nt","prot_aa","solute_atoms",
            "interface_res","nmr_fluct_A","qualifies","selected","split"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda r: (-r["selected"], r["score"])):
            w.writerow({c: r[c] for c in cols})

    n_sel = sum(r["selected"] for r in rows)
    n_nmr_sel = sum(1 for r in rows if r["selected"] and r["method"] == "NMR")
    print(f"scanned {len(rows)} complexes -> {out}")
    print(f"selected {n_sel} MD candidates ({n_nmr_sel} NMR), "
          f"train={sorted(train_ids)}")
    print(f"{'ID':6s} {'meth':6s} {'mdl':>4s} {'nt':>4s} {'aa':>4s} {'atoms':>6s} "
          f"{'iface':>5s} {'fluctA':>6s}  split")
    for r in sorted(rows, key=lambda r: (-r["selected"], r["score"])):
        if r["selected"]:
            print(f"{r['pdb_id']:6s} {r['method']:6s} {r['n_models']:4d} "
                  f"{r['rna_nt']:4d} {r['prot_aa']:4d} {r['solute_atoms']:6d} "
                  f"{r['interface_res']:5d} {r['nmr_fluct_A']:6.2f}  {r['split']}")

if __name__ == "__main__":
    main()
