#!/usr/bin/env python3
"""drpscore_parse.py — parse DRPScore scoring_*.txt into one tidy CSV.

Each scoring.txt line is a python literal: ('<path>.npy', [[p0, p1]]).
Output: results/drpscore/drpscore_scores.csv with
  tag, system, frame, p0, p1, p_native
p_native defaults to p1 (README: "third column = probability of native");
VERIFY against MD frames (should score high) before trusting — flip with
--flip if MD comes out high on p0 instead.
"""
import argparse, ast, os, glob
import pandas as pd

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRP = os.path.join(ROOT, "results", "drpscore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", action="store_true", help="native = p0 instead of p1")
    args = ap.parse_args()

    rows = []
    for txt in sorted(glob.glob(os.path.join(DRP, "scoring_*.txt"))):
        tag = os.path.basename(txt)[len("scoring_"):-len(".txt")]
        for line in open(txt):
            line = line.strip()
            if not line:
                continue
            try:
                name, probs = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            p0, p1 = probs[0]
            base = os.path.basename(name).replace(".pdb.npy", "").replace(".pdb", "")
            # <ID>_f<frame> convention from drpscore_prep.py
            if "_f" in base:
                pid, fr = base.rsplit("_f", 1)
            else:
                pid, fr = base, ""
            rows.append(dict(tag=tag, system=pid, frame=fr,
                             p0=p0, p1=p1,
                             p_native=p0 if args.flip else p1))
    df = pd.DataFrame(rows)
    out = os.path.join(DRP, "drpscore_scores.csv")
    df.to_csv(out, index=False)
    print(f"[parse] {len(df)} scores -> {out}")
    if len(df):
        print(df.groupby("tag")["p_native"].agg(["count", "mean", "std"]).round(3))


if __name__ == "__main__":
    main()
