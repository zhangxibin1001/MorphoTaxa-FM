#!/usr/bin/env python
from __future__ import annotations

import argparse, csv
from pathlib import Path


def read_names(path: Path, column: str) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {row[column].strip().lower() for row in csv.DictReader(f) if row.get(column, "").strip()}


def main():
    ap = argparse.ArgumentParser(description="Build shared-species list between two taxonomy/class-mapping CSVs.")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--column", default="scientific_name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shared = sorted(read_names(Path(args.a), args.column) & read_names(Path(args.b), args.column))
    Path(args.out).write_text("\n".join(shared) + ("\n" if shared else ""), encoding="utf-8")
    print(f"shared_species={len(shared)}")

if __name__ == "__main__":
    main()
