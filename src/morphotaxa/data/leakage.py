from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Leak:
    sha256: str
    splits: tuple[str, ...]
    paths: tuple[str, ...]


def exact_cross_split_leaks(rows) -> list[Leak]:
    by_hash = defaultdict(list)
    for row in rows:
        if row.sha256:
            by_hash[row.sha256].append(row)
    leaks = []
    for h, items in by_hash.items():
        splits = sorted({x.split for x in items})
        if len(splits) > 1:
            leaks.append(Leak(h, tuple(splits), tuple(x.relpath for x in items)))
    return leaks
