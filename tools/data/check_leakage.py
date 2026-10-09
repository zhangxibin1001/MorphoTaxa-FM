#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Check exact SHA256 overlap across splits from a generated manifest.csv")
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    by_hash = defaultdict(list)
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sha256"):
                by_hash[row["sha256"]].append(row)
    leaks = []
    for h, items in by_hash.items():
        splits = sorted({x["split"] for x in items})
        if len(splits) > 1:
            leaks.append({"sha256": h, "splits": splits, "paths": [x["relpath"] for x in items]})
    print(json.dumps({"cross_split_duplicate_groups": len(leaks), "leaks": leaks[:100]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
