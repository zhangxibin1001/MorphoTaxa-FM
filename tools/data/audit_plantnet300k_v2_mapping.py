from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/root/autodl-tmp/datasets_raw/plantnet300kV2"
)

IMAGE_META = ROOT / "plantnet300K_metadata.csv"
SPECIES_META = ROOT / "species_metadata.csv"

OUT_DIR = Path(
    "data_protocols/plantnet300k_v2"
)

OUT_MAPPING = (
    OUT_DIR
    / "class_idx_to_species_id_v2.json"
)

EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

EXPECTED_SPLITS = {
    "train": 243866,
    "val": 31115,
    "test": 31106,
}


def main():
    print("===== PLANTNET-300K V2 MAPPING AUDIT =====")

    assert ROOT.exists()
    assert IMAGE_META.exists()
    assert SPECIES_META.exists()

    # -------------------------------------------------
    # 1. Current V2 image metadata
    # -------------------------------------------------
    meta = pd.read_csv(
        IMAGE_META,
        low_memory=False,
    )

    required = {
        "species_id",
        "split",
        "PN_hash",
    }

    assert required.issubset(meta.columns)

    meta["species_id"] = (
        meta["species_id"]
        .astype(int)
    )

    print("\n===== CURRENT METADATA =====")
    print("rows =", len(meta))

    assert len(meta) == 306087

    split_counts = (
        meta["split"]
        .value_counts()
        .to_dict()
    )

    print("split counts =", split_counts)

    for split, expected in EXPECTED_SPLITS.items():
        assert int(
            split_counts.get(split, 0)
        ) == expected

    active_species = sorted(
        meta["species_id"]
        .unique()
        .tolist()
    )

    print(
        "active species IDs =",
        len(active_species),
    )

    assert len(active_species) == 1000, (
        "Current V2 metadata itself must "
        "contain exactly 1000 active species."
    )

    print(
        "species id min/max =",
        min(active_species),
        max(active_species),
    )

    # -------------------------------------------------
    # 2. Species-name metadata
    # -------------------------------------------------
    species = pd.read_csv(
        SPECIES_META,
        low_memory=False,
    )

    assert "species_id" in species.columns
    assert "full_species" in species.columns

    species["species_id"] = (
        species["species_id"]
        .astype(int)
    )

    species_name = {}

    for row in species.itertuples(index=False):
        sid = int(row.species_id)

        name = str(
            row.full_species
        ).strip()

        if sid in species_name:
            raise RuntimeError(
                f"Duplicate species_id "
                f"in species_metadata: {sid}"
            )

        species_name[sid] = name

    missing_names = [
        sid
        for sid in active_species
        if sid not in species_name
        or not species_name[sid]
        or species_name[sid].lower() == "nan"
    ]

    print(
        "active species missing names =",
        len(missing_names),
    )

    assert not missing_names

    # -------------------------------------------------
    # 3. Index every actual V2 image:
    #    (split, PN_hash) -> folder class index
    # -------------------------------------------------
    print("\n===== INDEX IMAGE FILES =====")

    file_index = {}
    folder_counts = defaultdict(int)

    expected_folders = {
        f"{i:04d}"
        for i in range(1000)
    }

    for split in [
        "train",
        "val",
        "test",
    ]:
        split_root = ROOT / split

        folders = {
            p.name
            for p in split_root.iterdir()
            if p.is_dir()
        }

        assert folders == expected_folders, (
            f"{split}: folder set mismatch"
        )

        for folder_name in sorted(folders):
            folder = split_root / folder_name
            class_idx = int(folder_name)

            for p in folder.iterdir():
                if (
                    not p.is_file()
                    or p.suffix.lower() not in EXTS
                ):
                    continue

                key = (
                    split,
                    p.stem,
                )

                if key in file_index:
                    raise RuntimeError(
                        f"Duplicate image hash key: {key}"
                    )

                file_index[key] = class_idx
                folder_counts[
                    (split, class_idx)
                ] += 1

    print(
        "indexed image files =",
        len(file_index),
    )

    assert len(file_index) == 306087

    # -------------------------------------------------
    # 4. Join CURRENT V2 metadata to ACTUAL files.
    #
    # Derive:
    # folder class index -> species_id
    #
    # No use of the 1081-class legacy file.
    # -------------------------------------------------
    print("\n===== DERIVE V2 CLASS MAPPING =====")

    folder_to_species_sets = {
        i: set()
        for i in range(1000)
    }

    missing_file_matches = []
    metadata_folder_mismatch = 0

    for row in meta.itertuples(index=False):
        split = str(row.split)
        stem = str(row.PN_hash)
        sid = int(row.species_id)

        key = (
            split,
            stem,
        )

        class_idx = file_index.get(key)

        if class_idx is None:
            if len(missing_file_matches) < 20:
                missing_file_matches.append(
                    (split, stem, sid)
                )
            continue

        folder_to_species_sets[
            class_idx
        ].add(sid)

    print(
        "missing metadata->file matches =",
        len(missing_file_matches),
    )

    if missing_file_matches:
        print(
            "examples =",
            missing_file_matches[:10],
        )

    assert not missing_file_matches, (
        "Some metadata hashes could not be "
        "matched to actual V2 image files."
    )

    mapping = {}

    ambiguous = {}

    for idx in range(1000):
        sids = (
            folder_to_species_sets[idx]
        )

        if len(sids) != 1:
            ambiguous[idx] = sorted(sids)
        else:
            mapping[idx] = next(
                iter(sids)
            )

    print(
        "unambiguous folders =",
        len(mapping),
    )

    print(
        "ambiguous/empty folders =",
        len(ambiguous),
    )

    if ambiguous:
        for idx, sids in list(
            ambiguous.items()
        )[:20]:
            print(
                f"{idx:04d}",
                "->",
                sids[:20],
            )

    assert not ambiguous, (
        "Each folder must map to exactly "
        "one species_id."
    )

    # Mapping must be bijective.
    mapped_species = list(
        mapping.values()
    )

    assert len(set(mapped_species)) == 1000

    assert set(mapped_species) == set(
        active_species
    )

    # -------------------------------------------------
    # 5. Report whether V2 happens to be identity.
    #    This is OBSERVED, not assumed.
    # -------------------------------------------------
    identity_count = sum(
        1
        for idx, sid in mapping.items()
        if idx == sid
    )

    print(
        "folder_idx == species_id:",
        identity_count,
        "/ 1000",
    )

    print("\n===== VERIFIED EXAMPLES =====")

    for idx in [
        0, 1, 2, 3, 4,
        995, 996, 997, 998, 999,
    ]:
        sid = mapping[idx]

        print(
            f"{idx:04d}",
            "-> species_id",
            sid,
            "->",
            species_name[sid],
        )

    # -------------------------------------------------
    # 6. Save a V2-specific verified mapping.
    # -------------------------------------------------
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_MAPPING.write_text(
        json.dumps(
            {
                f"{idx:04d}": {
                    "class_idx": idx,
                    "species_id": mapping[idx],
                    "full_species":
                        species_name[
                            mapping[idx]
                        ],
                }
                for idx in range(1000)
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("PLANTNET V2 ACTIVE SPECIES: PASS")
    print("PLANTNET V2 FILE JOIN: PASS")
    print("PLANTNET V2 CLASS MAPPING: PASS")
    print("PLANTNET V2 SPECIES NAMES: PASS")
    print(
        "saved:",
        OUT_MAPPING,
    )


if __name__ == "__main__":
    main()
