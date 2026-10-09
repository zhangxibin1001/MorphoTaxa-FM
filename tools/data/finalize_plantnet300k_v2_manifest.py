from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/datasets_raw/plantnet300kV2"
)

PROTO = Path(
    "data_protocols/plantnet300k_v2"
)

MANIFEST = PROTO / "manifest.csv"
CLASSES = PROTO / "classes.csv"
PROTOCOL = PROTO / "protocol.json"
LONGTAIL = PROTO / "longtail_bins.json"

V2_MAPPING = (
    PROTO
    / "class_idx_to_species_id_v2.json"
)

IMAGE_META = (
    ROOT
    / "plantnet300K_metadata.csv"
)

SPECIES_META = (
    ROOT
    / "species_metadata.csv"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():
    assert MANIFEST.exists()
    assert V2_MAPPING.exists()
    assert IMAGE_META.exists()
    assert SPECIES_META.exists()

    mapping_raw = json.loads(
        V2_MAPPING.read_text(
            encoding="utf-8"
        )
    )

    assert len(mapping_raw) == 1000

    mapping = {}

    for i in range(1000):
        key = f"{i:04d}"

        assert key in mapping_raw

        item = mapping_raw[key]

        assert int(
            item["class_idx"]
        ) == i

        assert int(
            item["species_id"]
        ) == i

        name = str(
            item["full_species"]
        ).strip()

        assert name
        assert name.lower() != "nan"

        mapping[i] = {
            "class_id": key,
            "species_id": i,
            "class_name": name,
        }

    # ---------------------------------
    # Rewrite manifest atomically
    # ---------------------------------
    tmp = MANIFEST.with_suffix(
        ".csv.tmp"
    )

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    row_count = 0

    with MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as src, tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as dst:

        reader = csv.DictReader(src)

        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "path",
                "split",
                "label",
                "class_name",
                "class_id",
                "species_id",
            ],
        )

        writer.writeheader()

        for row in reader:
            label = int(
                row["label"]
            )

            assert 0 <= label < 1000

            info = mapping[label]

            # Temporary manifest was already
            # generated from folders 0000..0999.
            assert (
                row["class_id"]
                == info["class_id"]
            )

            split = row["split"]

            assert split in split_counts

            writer.writerow({
                "path":
                    row["path"],
                "split":
                    split,
                "label":
                    label,
                "class_name":
                    info["class_name"],
                "class_id":
                    info["class_id"],
                "species_id":
                    info["species_id"],
            })

            split_counts[split] += 1
            row_count += 1

    assert row_count == 306087

    assert split_counts == {
        "train": 243866,
        "val": 31115,
        "test": 31106,
    }

    tmp.replace(
        MANIFEST
    )

    # ---------------------------------
    # Final classes.csv
    # ---------------------------------
    with CLASSES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "label",
            "class_id",
            "species_id",
            "class_name",
        ])

        for i in range(1000):
            info = mapping[i]

            writer.writerow([
                i,
                info["class_id"],
                info["species_id"],
                info["class_name"],
            ])

    # ---------------------------------
    # Protocol
    # ---------------------------------
    if PROTOCOL.exists():
        protocol = json.loads(
            PROTOCOL.read_text(
                encoding="utf-8"
            )
        )
    else:
        protocol = {}

    protocol.update({
        "dataset":
            "PlantNet-300K V2",
        "protocol":
            "PlantNet-300K-V2-Official-2026",
        "num_classes":
            1000,
        "counts": {
            "train": 243866,
            "val": 31115,
            "test": 31106,
        },
        "total":
            306087,
        "split_policy":
            "official train/val/test; no re-split",
        "class_index_policy":
            (
                "folder indices 0000..0999 "
                "verified against all 306087 "
                "metadata records; "
                "folder_idx == species_id"
            ),
        "semantic_names":
            (
                "full_species from "
                "species_metadata.csv"
            ),
        "mapping_source":
            (
                "derived and verified using "
                "PN_hash + split + actual image "
                "folder + plantnet300K_metadata.csv"
            ),
        "legacy_mapping_policy":
            (
                "metadata_v1_legacy 1081-class "
                "mapping excluded from V2 protocol"
            ),
        "test_policy":
            (
                "locked; no model selection "
                "or hyperparameter tuning on test"
            ),
        "longtail_policy": {
            "source":
                "train split only",
            "head":
                "n > 100",
            "medium":
                "20 < n <= 100",
            "few":
                "n <= 20",
        },
    })

    PROTOCOL.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------
    # Final freeze
    # ---------------------------------
    freeze = (
        PROTO
        / "freeze_final.sha256"
    )

    freeze_files = [
        MANIFEST,
        CLASSES,
        PROTOCOL,
        LONGTAIL,
        V2_MAPPING,
        IMAGE_META,
        SPECIES_META,
    ]

    with freeze.open(
        "w",
        encoding="utf-8",
    ) as f:
        for p in freeze_files:
            assert p.exists()

            f.write(
                f"{sha256(p)}  {p}\n"
            )

    print(
        "manifest rows =",
        row_count,
    )

    print(
        "split counts =",
        split_counts,
    )

    print("\nfirst 5:")

    for i in range(5):
        print(
            i,
            mapping[i]["class_name"],
        )

    print("\nlast 5:")

    for i in range(
        995,
        1000,
    ):
        print(
            i,
            mapping[i]["class_name"],
        )

    print()
    print(
        "PLANTNET V2 FINAL MANIFEST: PASS"
    )
    print(
        "PLANTNET V2 FINAL FREEZE: PASS"
    )
    print(
        "freeze =",
        freeze,
    )


if __name__ == "__main__":
    main()
