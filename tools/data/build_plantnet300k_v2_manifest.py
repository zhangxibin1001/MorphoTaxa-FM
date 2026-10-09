from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


EXPECTED = {
    "train": 243866,
    "val": 31115,
    "test": 31106,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    assert root.exists()

    split_classes = {}

    for split in ["train", "val", "test"]:
        sp = root / split

        if not sp.exists():
            raise RuntimeError(
                f"Missing official split: {sp}"
            )

        split_classes[split] = {
            p.name
            for p in sp.iterdir()
            if p.is_dir()
        }

    union_classes = sorted(
        set().union(
            *split_classes.values()
        )
    )

    if len(union_classes) != 1000:
        raise RuntimeError(
            f"Expected 1000 classes, "
            f"found {len(union_classes)}"
        )

    # Stable index determined once from
    # complete official class ID union.
    class_to_idx = {
        name: i
        for i, name
        in enumerate(union_classes)
    }

    rows = []
    counts = {}

    for split in [
        "train",
        "val",
        "test",
    ]:
        sp = root / split

        files = sorted([
            p
            for p in sp.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTS
        ])

        counts[split] = len(files)

        for p in files:
            rel = p.relative_to(root)

            parts = rel.parts

            if len(parts) < 3:
                raise RuntimeError(
                    f"Unexpected path: {rel}"
                )

            class_id = parts[1]

            if class_id not in class_to_idx:
                raise RuntimeError(
                    f"Unknown class: {class_id}"
                )

            rows.append({
                "path": str(rel),
                "split": split,
                "label":
                    class_to_idx[class_id],
                # Temporary semantic name.
                # This MUST later be replaced
                # from official metadata.
                "class_name": class_id,
                "class_id": class_id,
            })

    print("counts =", counts)

    for split, expected in EXPECTED.items():
        if counts[split] != expected:
            raise RuntimeError(
                f"{split}: "
                f"{counts[split]} != "
                f"expected {expected}"
            )

    if len(rows) != 306087:
        raise RuntimeError(
            f"total={len(rows)} "
            "!=306087"
        )

    manifest = out / "manifest.csv"

    with manifest.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "split",
                "label",
                "class_name",
                "class_id",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    classes = out / "classes.csv"

    with classes.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "label",
            "class_id",
            "class_name",
        ])

        for cid, label in sorted(
            class_to_idx.items(),
            key=lambda x: x[1],
        ):
            writer.writerow([
                label,
                cid,
                cid,
            ])

    protocol = {
        "dataset":
            "PlantNet-300K V2",
        "protocol":
            "PlantNet-300K-V2-Official-2026",
        "num_classes":
            len(union_classes),
        "split_policy":
            "official train/val/test; no re-split",
        "counts":
            counts,
        "total":
            len(rows),
        "semantic_names":
            "PENDING_OFFICIAL_METADATA_MAPPING",
    }

    protocol_path = (
        out / "protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    freeze = out / "freeze.sha256"

    with freeze.open(
        "w",
        encoding="utf-8",
    ) as f:
        for p in [
            manifest,
            classes,
            protocol_path,
        ]:
            f.write(
                f"{sha256(p)}  {p.name}\n"
            )

    print()
    print("PASS")
    print(
        "manifest:",
        manifest,
    )
    print(
        "classes:",
        classes,
    )
    print(
        "protocol:",
        protocol_path,
    )
    print(
        "freeze:",
        freeze,
    )


if __name__ == "__main__":
    main()
