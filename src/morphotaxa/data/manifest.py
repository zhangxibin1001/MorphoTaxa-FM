from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import csv
import hashlib

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestRow:
    relpath: str
    split: str
    class_name: str
    class_id: int
    size_bytes: int
    sha256: str


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_folder_manifest(root: Path, splits: Iterable[str] = ("train", "val", "test"), hash_files: bool = True) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
        class_to_id = {name: i for i, name in enumerate(classes)}
        for class_name in classes:
            for p in sorted((split_dir / class_name).rglob("*")):
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                    continue
                rows.append(ManifestRow(
                    relpath=str(p.relative_to(root)),
                    split=split,
                    class_name=class_name,
                    class_id=class_to_id[class_name],
                    size_bytes=p.stat().st_size,
                    sha256=file_sha256(p) if hash_files else "",
                ))
    return rows


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ["relpath", "split", "class_name", "class_id", "size_bytes", "sha256"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
