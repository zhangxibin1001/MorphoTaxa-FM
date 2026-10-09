#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

from morphotaxa.data.manifest import build_folder_manifest, write_manifest
from morphotaxa.data.leakage import exact_cross_split_leaks


def main():
    ap = argparse.ArgumentParser(description="Audit a folder-style train/val/test dataset and emit a reproducible manifest.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--expected-classes", type=int, default=None)
    ap.add_argument("--out", default="outputs/audit")
    ap.add_argument("--no-hash", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise SystemExit(
            f"DATASET ROOT NOT FOUND: {root}"
        )

    if not root.is_dir():
        raise SystemExit(
            f"DATASET ROOT IS NOT A DIRECTORY: {root}"
        )

    rows = build_folder_manifest(
        root,
        hash_files=not args.no_hash
    )
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, out / "manifest.csv")

    split_counts = Counter(r.split for r in rows)
    split_classes = defaultdict(set)
    for r in rows:
        split_classes[r.split].add(r.class_name)
    class_union = sorted(set().union(*split_classes.values())) if split_classes else []
    leaks = exact_cross_split_leaks(rows) if not args.no_hash else []
    report = {
        "name": args.name,
        "root": str(root),
        "num_images": len(rows),
        "num_classes_union": len(class_union),
        "expected_classes": args.expected_classes,
        "split_counts": dict(split_counts),
        "classes_per_split": {k: len(v) for k, v in split_classes.items()},
        "exact_cross_split_leaks": len(leaks),
        "status": "PASS",
        "warnings": [],
    }
    if len(rows) == 0:
        report["status"] = "FAIL"
        report["warnings"].append(
            "dataset contains zero recognized images"
        )

    if args.expected_classes is not None and len(class_union) != args.expected_classes:
        report["status"] = "REVIEW"
        report["warnings"].append(f"class count {len(class_union)} != expected {args.expected_classes}")
    if leaks:
        report["status"] = "REVIEW"
        report["warnings"].append(f"found {len(leaks)} exact cross-split duplicate groups")
        (out / "exact_leaks.json").write_text(json.dumps([l.__dict__ for l in leaks], indent=2), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
