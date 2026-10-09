#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+", help="metrics.json files from independent seeds")
    args = ap.parse_args()
    rows = [json.loads(Path(p).read_text()) for p in args.metrics]
    keys = sorted(set.intersection(*(set(r) for r in rows)))
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        if len(vals) == len(rows):
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
