from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


MANIFEST = Path(
    "data_protocols/plantnet300k_v2/"
    "manifest.csv"
)

OUT = Path(
    "data_protocols/plantnet300k_v2/"
    "longtail_bins.json"
)


train_counts = Counter()

with MANIFEST.open(
    "r",
    encoding="utf-8",
) as f:
    for row in csv.DictReader(f):
        if row["split"] != "train":
            continue

        train_counts[
            int(row["label"])
        ] += 1


bins = {
    "head": [],
    "medium": [],
    "few": [],
}

records = {}

for label in range(1000):
    n = train_counts[label]

    if n > 100:
        group = "head"

    elif n > 20:
        group = "medium"

    else:
        group = "few"

    bins[group].append(label)

    records[str(label)] = {
        "train_count": n,
        "group": group,
    }


obj = {
    "definition": {
        "head":
            "train_count > 100",
        "medium":
            "20 < train_count <= 100",
        "few":
            "train_count <= 20",
    },
    "num_classes": {
        k: len(v)
        for k, v in bins.items()
    },
    "classes": bins,
    "records": records,
}


OUT.write_text(
    json.dumps(
        obj,
        indent=2,
    ),
    encoding="utf-8",
)


print("===== LONG-TAIL BINS =====")

for k in [
    "head",
    "medium",
    "few",
]:
    print(
        k,
        "=",
        len(bins[k]),
    )

print("saved:", OUT)
