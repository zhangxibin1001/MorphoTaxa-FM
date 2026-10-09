#!/usr/bin/env python
from __future__ import annotations

import argparse, json
import numpy as np
from sklearn.metrics import f1_score


def main():
    ap = argparse.ArgumentParser(description="Paired bootstrap CI for macro-F1 difference between two prediction files.")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--pred-a", required=True)
    ap.add_argument("--pred-b", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    y, a, b = map(np.load, [args.targets, args.pred_a, args.pred_b])
    if not (len(y)==len(a)==len(b)):
        raise ValueError("arrays must have equal length")
    rng = np.random.default_rng(args.seed)
    d=[]
    for _ in range(args.n):
        idx=rng.integers(0,len(y),len(y))
        d.append(f1_score(y[idx], b[idx], average="macro", zero_division=0)-f1_score(y[idx], a[idx], average="macro", zero_division=0))
    lo, hi=np.percentile(d,[2.5,97.5])
    print(json.dumps({"mean_delta_macro_f1": float(np.mean(d)), "ci95": [float(lo), float(hi)], "bootstrap_n": args.n}, indent=2))

if __name__ == "__main__":
    main()
