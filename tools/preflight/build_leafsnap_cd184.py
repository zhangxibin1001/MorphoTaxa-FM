from pathlib import Path
from collections import Counter
import argparse
import json
import random
import re

import pandas as pd


def norm_species(x):
    return str(x).strip().lower().replace(" ", "_")


def stem_candidate(path):
    """
    ny1157-01-1.jpg -> ny1157-01
    ny1157-01-2.jpg -> ny1157-01

    对没有这种尾部编号结构的文件，暂时返回完整 stem。
    """
    stem = Path(path).stem
    m = re.match(r"^(.*)-(\d+)$", stem)
    return m.group(1) if m else stem


def group_split(df, val_ratio, seed):
    """
    每个 species 内按 group 划分，而不是按 image 划分。
    同一个 group 永远不会同时出现在 train / val。
    """
    rng = random.Random(seed)
    train_parts = []
    val_parts = []

    for species, sub in df.groupby("species", sort=True):
        groups = sorted(sub["group_id"].unique())
        rng.shuffle(groups)

        n = len(groups)

        if n <= 1:
            # 无法在不泄漏的前提下切分
            train_groups = groups
            val_groups = []
        else:
            n_val = max(1, round(n * val_ratio))
            n_val = min(n_val, n - 1)

            val_groups = set(groups[:n_val])
            train_groups = set(groups[n_val:])

            train_parts.append(
                sub[sub["group_id"].isin(train_groups)]
            )
            val_parts.append(
                sub[sub["group_id"].isin(val_groups)]
            )
            continue

        train_parts.append(sub)

    train_df = pd.concat(train_parts, ignore_index=True)

    if val_parts:
        val_df = pd.concat(val_parts, ignore_index=True)
    else:
        val_df = df.iloc[0:0].copy()

    return train_df, val_df


def overlap(a, b):
    return sorted(
        set(a["group_id"]) & set(b["group_id"])
    )


def save_csv(df, path):
    df.sort_values(
        ["label", "group_id", "image_path"]
    ).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/leafsnap"),
    )

    ap.add_argument(
        "--metadata",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data_protocols/leafsnap"),
    )

    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.metadata is None:
        candidates = [
            root / "leafsnap-dataset-images.txt",
            root / "leafsnap-dataset" / "leafsnap-dataset-images.txt",
        ]

        metadata = None
        for p in candidates:
            if p.exists():
                metadata = p
                break

        if metadata is None:
            found = list(
                root.rglob("leafsnap-dataset-images.txt")
            )
            if len(found) == 1:
                metadata = found[0]
            elif len(found) == 0:
                raise FileNotFoundError(
                    f"Cannot find leafsnap-dataset-images.txt under {root}"
                )
            else:
                raise RuntimeError(
                    "Multiple metadata files found:\n"
                    + "\n".join(map(str, found))
                )
    else:
        metadata = args.metadata.resolve()

    print("===== LEAFSNAP-CD-184 BUILDER =====")
    print("root     =", root)
    print("metadata =", metadata)
    print("out      =", out)
    print("seed     =", args.seed)

    df = pd.read_csv(metadata, sep="\t")

    required = {
        "file_id",
        "image_path",
        "segmented_path",
        "species",
        "source",
    }

    missing_cols = required - set(df.columns)
    assert not missing_cols, missing_cols

    df["source"] = (
        df["source"].astype(str).str.strip().str.lower()
    )
    df["species"] = df["species"].astype(str).str.strip()

    lab_species = set(
        df.loc[df.source == "lab", "species"]
    )
    field_species = set(
        df.loc[df.source == "field", "species"]
    )

    common = sorted(lab_species & field_species)

    print()
    print("===== CLASS ALIGNMENT =====")
    print("lab species   =", len(lab_species))
    print("field species =", len(field_species))
    print("common        =", len(common))
    print("excluded lab-only =", sorted(lab_species - field_species))

    assert len(common) == 184, (
        f"Expected 184 common species, got {len(common)}"
    )

    df = df[df["species"].isin(common)].copy()

    class_to_idx = {
        species: idx
        for idx, species in enumerate(common)
    }

    df["label"] = df["species"].map(class_to_idx)
    df["species_key"] = df["species"].map(norm_species)

    # ---------------------------------------------------------
    # Absolute path validation
    # ---------------------------------------------------------
    df["absolute_path"] = df["image_path"].map(
        lambda x: str((root / str(x)).resolve())
    )

    exists = df["absolute_path"].map(
        lambda x: Path(x).is_file()
    )

    if not exists.all():
        bad = df.loc[
            ~exists,
            ["file_id", "image_path", "absolute_path"]
        ]

        print(bad.head(20).to_string(index=False))
        raise FileNotFoundError(
            f"{len(bad)} metadata image paths do not exist"
        )

    # ---------------------------------------------------------
    # Conservative specimen/group inference
    #
    # First:
    #   ny1157-01-1 -> ny1157-01
    #
    # Only use this shorter candidate as a group if multiple
    # images from the same species/source share it.
    # Otherwise retain the full stem as an individual group.
    # ---------------------------------------------------------
    df["group_candidate"] = df["image_path"].map(
        stem_candidate
    )

    candidate_counts = (
        df.groupby(
            ["source", "species", "group_candidate"]
        )
        .size()
        .rename("candidate_n")
        .reset_index()
    )

    df = df.merge(
        candidate_counts,
        on=["source", "species", "group_candidate"],
        how="left",
    )

    full_stem = df["image_path"].map(
        lambda x: Path(x).stem
    )

    group_core = df["group_candidate"].where(
        df["candidate_n"] >= 2,
        full_stem,
    )

    # Species is included deliberately so identical strings from
    # different species cannot accidentally merge.
    df["group_id"] = (
        df["species_key"]
        + "::"
        + group_core.astype(str)
    )

    # ---------------------------------------------------------
    # Domain subsets
    # ---------------------------------------------------------
    lab = df[df.source == "lab"].copy()
    field = df[df.source == "field"].copy()

    print()
    print("===== COMMON DOMAIN COUNTS =====")
    print("lab images   =", len(lab))
    print("field images =", len(field))
    print("total        =", len(df))

    assert len(lab) == 23027, len(lab)
    assert len(field) == 7719, len(field)
    assert len(df) == 30746, len(df)

    # ---------------------------------------------------------
    # Group diagnostics
    # ---------------------------------------------------------
    print()
    print("===== GROUP DIAGNOSTICS =====")

    for source_name, source_df in [
        ("lab", lab),
        ("field", field),
    ]:
        group_per_class = (
            source_df.groupby("species")["group_id"]
            .nunique()
        )

        print(
            source_name,
            "groups =",
            source_df["group_id"].nunique(),
            "groups/class min =",
            int(group_per_class.min()),
            "median =",
            float(group_per_class.median()),
            "max =",
            int(group_per_class.max()),
        )

        weak = group_per_class[
            group_per_class < 2
        ]

        if len(weak):
            print(
                f"WARNING: {source_name} species with <2 groups:"
            )
            print(weak.to_string())

    # ---------------------------------------------------------
    # Group-aware within-domain splits
    # ---------------------------------------------------------
    lab_train, lab_val = group_split(
        lab,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    field_train, field_val = group_split(
        field,
        val_ratio=args.val_ratio,
        seed=args.seed + 1,
    )

    lab_overlap = overlap(lab_train, lab_val)
    field_overlap = overlap(field_train, field_val)

    assert not lab_overlap, (
        f"LAB GROUP LEAKAGE: {lab_overlap[:20]}"
    )

    assert not field_overlap, (
        f"FIELD GROUP LEAKAGE: {field_overlap[:20]}"
    )

    # ---------------------------------------------------------
    # Save classes
    # ---------------------------------------------------------
    classes_path = out / "classes_184.txt"

    with classes_path.open("w", encoding="utf-8") as f:
        for idx, species in enumerate(common):
            f.write(
                f"{idx}\t{species}\t{norm_species(species)}\n"
            )

    # ---------------------------------------------------------
    # Save CSV files
    # ---------------------------------------------------------
    save_csv(lab, out / "lab_all.csv")
    save_csv(field, out / "field_all.csv")

    save_csv(
        lab_train,
        out / "lab_train.csv",
    )
    save_csv(
        lab_val,
        out / "lab_val.csv",
    )

    save_csv(
        field_train,
        out / "field_train.csv",
    )
    save_csv(
        field_val,
        out / "field_val.csv",
    )

    # ---------------------------------------------------------
    # Protocol JSON
    # ---------------------------------------------------------
    lab_to_field = {
        "name": "LeafSnap-CD-184 Lab-to-Field",
        "num_classes": 184,
        "seed": args.seed,
        "source_domain": "lab",
        "target_domain": "field",
        "train_csv": "lab_train.csv",
        "validation_csv": "lab_val.csv",
        "test_csv": "field_all.csv",
        "checkpoint_selection": "lab_val",
        "target_domain_used_for_training": False,
        "group_aware_source_split": True,
    }

    field_to_lab = {
        "name": "LeafSnap-CD-184 Field-to-Lab",
        "num_classes": 184,
        "seed": args.seed,
        "source_domain": "field",
        "target_domain": "lab",
        "train_csv": "field_train.csv",
        "validation_csv": "field_val.csv",
        "test_csv": "lab_all.csv",
        "checkpoint_selection": "field_val",
        "target_domain_used_for_training": False,
        "group_aware_source_split": True,
    }

    with (
        out / "protocol_lab_to_field.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            lab_to_field,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (
        out / "protocol_field_to_lab.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            field_to_lab,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Detailed split audit
    # ---------------------------------------------------------
    audit = {
        "protocol": "LeafSnap-CD-184",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "num_classes": len(common),

        "images": {
            "lab_all": len(lab),
            "field_all": len(field),
            "lab_train": len(lab_train),
            "lab_val": len(lab_val),
            "field_train": len(field_train),
            "field_val": len(field_val),
        },

        "groups": {
            "lab_all": int(
                lab["group_id"].nunique()
            ),
            "field_all": int(
                field["group_id"].nunique()
            ),
            "lab_train": int(
                lab_train["group_id"].nunique()
            ),
            "lab_val": int(
                lab_val["group_id"].nunique()
            ),
            "field_train": int(
                field_train["group_id"].nunique()
            ),
            "field_val": int(
                field_val["group_id"].nunique()
            ),
        },

        "class_coverage": {
            "lab_train": int(
                lab_train["species"].nunique()
            ),
            "lab_val": int(
                lab_val["species"].nunique()
            ),
            "field_train": int(
                field_train["species"].nunique()
            ),
            "field_val": int(
                field_val["species"].nunique()
            ),
        },

        "leakage": {
            "lab_train_val_group_overlap":
                len(lab_overlap),

            "field_train_val_group_overlap":
                len(field_overlap),
        },

        "excluded_species": sorted(
            lab_species ^ field_species
        ),
    }

    with (
        out / "split_audit.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("===== SPLIT COUNTS =====")
    print("lab_train   =", len(lab_train))
    print("lab_val     =", len(lab_val))
    print("field_train =", len(field_train))
    print("field_val   =", len(field_val))

    print()
    print("===== CLASS COVERAGE =====")
    print(
        "lab_train   =",
        lab_train.species.nunique(),
    )
    print(
        "lab_val     =",
        lab_val.species.nunique(),
    )
    print(
        "field_train =",
        field_train.species.nunique(),
    )
    print(
        "field_val   =",
        field_val.species.nunique(),
    )

    print()
    print("===== LEAKAGE AUDIT =====")
    print(
        "lab train/val group overlap =",
        len(lab_overlap),
    )
    print(
        "field train/val group overlap =",
        len(field_overlap),
    )

    assert (
        lab_train.species.nunique() == 184
    ), "lab_train does not cover all 184 classes"

    assert (
        field_train.species.nunique() == 184
    ), "field_train does not cover all 184 classes"

    assert len(lab_overlap) == 0
    assert len(field_overlap) == 0

    print()
    print("LEAFSNAP-CD-184 BUILD: PASS")
    print("Outputs:", out)


if __name__ == "__main__":
    main()
