from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import csv
import re

ROOT = Path(
    "/root/autodl-tmp/datasets_raw/leafsnap"
)

IMAGE_ROOT = (
    ROOT / "dataset" / "images"
)

META = (
    ROOT / "leafsnap-dataset-images.txt"
)

README = (
    ROOT / "leafsnap-dataset-readme.txt"
)

EXTS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp",
    ".tif", ".tiff",
}


print(
    "===== LEAFSNAP PROTOCOL AUDIT ====="
)

assert IMAGE_ROOT.exists()
assert META.exists()
assert README.exists()


# ---------------------------------------------------------
# 1. Show README header
# ---------------------------------------------------------

print(
    "\n===== README HEAD ====="
)

readme_lines = README.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

for line in readme_lines[:80]:
    print(line)


# ---------------------------------------------------------
# 2. Show raw metadata header
# ---------------------------------------------------------

print(
    "\n===== METADATA HEAD ====="
)

meta_lines = META.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

for line in meta_lines[:20]:
    print(repr(line))


# ---------------------------------------------------------
# 3. Filesystem species/domain audit
# ---------------------------------------------------------

domain_species = {
    "lab": defaultdict(list),
    "field": defaultdict(list),
}

all_images = []

for domain in [
    "lab",
    "field",
]:
    droot = IMAGE_ROOT / domain

    assert droot.exists(), (
        f"missing domain dir: {droot}"
    )

    for species_dir in sorted(
        p for p in droot.iterdir()
        if p.is_dir()
    ):
        images = sorted(
            p
            for p in species_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in EXTS
            )
        )

        if not images:
            continue

        species = species_dir.name

        domain_species[
            domain
        ][species].extend(images)

        all_images.extend(
            (
                domain,
                species,
                p,
            )
            for p in images
        )


lab_species = set(
    domain_species["lab"]
)

field_species = set(
    domain_species["field"]
)

both_species = (
    lab_species
    & field_species
)

lab_only = (
    lab_species
    - field_species
)

field_only = (
    field_species
    - lab_species
)


print(
    "\n===== DOMAIN SPECIES ====="
)

print(
    "lab species =",
    len(lab_species),
)

print(
    "field species =",
    len(field_species),
)

print(
    "intersection =",
    len(both_species),
)

print(
    "lab only =",
    len(lab_only),
)

print(
    "field only =",
    len(field_only),
)


if lab_only:
    print(
        "\nLAB ONLY:"
    )

    for x in sorted(
        lab_only
    ):
        print(x)


if field_only:
    print(
        "\nFIELD ONLY:"
    )

    for x in sorted(
        field_only
    ):
        print(x)


# ---------------------------------------------------------
# 4. Counts per common species
# ---------------------------------------------------------

records = []

for species in sorted(
    both_species
):
    n_lab = len(
        domain_species[
            "lab"
        ][species]
    )

    n_field = len(
        domain_species[
            "field"
        ][species]
    )

    records.append(
        (
            species,
            n_lab,
            n_field,
        )
    )


print(
    "\n===== COMMON-SPECIES COUNTS ====="
)

print(
    "common species =",
    len(records)
)

print(
    "total lab common =",
    sum(
        x[1]
        for x in records
    ),
)

print(
    "total field common =",
    sum(
        x[2]
        for x in records
    ),
)


def summary(values):
    values = sorted(values)

    n = len(values)

    if n == 0:
        return {}

    return {
        "min": values[0],
        "median": values[
            n // 2
        ],
        "max": values[-1],
        "mean":
            sum(values) / n,
    }


print(
    "lab/class =",
    summary(
        [
            x[1]
            for x in records
        ]
    ),
)

print(
    "field/class =",
    summary(
        [
            x[2]
            for x in records
        ]
    ),
)


print(
    "\n===== LOW-SUPPORT SPECIES ====="
)

for species, n_lab, n_field in records:
    if (
        n_lab < 5
        or n_field < 2
    ):
        print(
            species,
            "lab=",
            n_lab,
            "field=",
            n_field,
        )


# ---------------------------------------------------------
# 5. Test possible eligibility rules
# ---------------------------------------------------------

rules = [
    (1, 1),
    (2, 1),
    (5, 1),
    (5, 2),
    (10, 2),
    (10, 3),
    (20, 3),
    (20, 5),
]

print(
    "\n===== ELIGIBILITY COUNTS ====="
)

for min_lab, min_field in rules:
    eligible = [
        x
        for x in records
        if (
            x[1] >= min_lab
            and x[2] >= min_field
        )
    ]

    print(
        f"lab>={min_lab}, "
        f"field>={min_field}:",
        len(eligible),
    )


# ---------------------------------------------------------
# 6. Filename/acquisition structure
# ---------------------------------------------------------

print(
    "\n===== FILENAME STRUCTURE ====="
)

examples = []

for (
    domain,
    species,
    path,
) in all_images[:1000]:

    stem = path.stem

    examples.append(
        (
            domain,
            species,
            stem,
            len(stem),
            stem.isdigit(),
        )
    )


length_counts = Counter(
    x[3]
    for x in examples
)

numeric_counts = Counter(
    x[4]
    for x in examples
)

print(
    "stem lengths =",
    dict(
        length_counts
    ),
)

print(
    "numeric =",
    dict(
        numeric_counts
    ),
)


# ---------------------------------------------------------
# 7. Prefix grouping diagnostics
#
# This is NOT declared an official specimen ID.
# We only inspect whether filename prefixes form repeated
# acquisition-like bundles.
# ---------------------------------------------------------

print(
    "\n===== PREFIX GROUP DIAGNOSTIC ====="
)

for prefix_len in [
    6,
    8,
    10,
    11,
    12,
]:
    groups = Counter()

    for (
        domain,
        species,
        path,
    ) in all_images:

        stem = path.stem

        if len(stem) >= prefix_len:
            key = (
                domain,
                species,
                stem[
                    :prefix_len
                ],
            )

            groups[key] += 1

    repeated = [
        n
        for n in groups.values()
        if n > 1
    ]

    print(
        "prefix_len =",
        prefix_len,
        "groups =",
        len(groups),
        "repeated_groups =",
        len(repeated),
        "images_in_repeated_groups =",
        sum(repeated),
        "max_group =",
        max(
            repeated,
            default=1,
        ),
    )


# ---------------------------------------------------------
# 8. Exact path metadata join reconnaissance
# ---------------------------------------------------------

print(
    "\n===== METADATA PATH RECON ====="
)

fs_relpaths = set()

for (
    domain,
    species,
    path,
) in all_images:
    fs_relpaths.add(
        str(
            path.relative_to(ROOT)
        ).replace(
            "\\",
            "/",
        )
    )


matched_lines = 0

for line in meta_lines:
    normalized = (
        line
        .replace("\\", "/")
        .strip()
    )

    if any(
        rel in normalized
        for rel in list(
            fs_relpaths
        )[:1000]
    ):
        matched_lines += 1


print(
    "metadata lines =",
    len(meta_lines),
)

print(
    "filesystem images =",
    len(all_images),
)

print(
    "sample path-containing "
    "metadata lines =",
    matched_lines,
)


# ---------------------------------------------------------
# 9. Overall exact counts
# ---------------------------------------------------------
print("\n===== METADATA PATH RECON =====")

metadata_records = meta_lines[1:]

print("metadata total lines =", len(meta_lines))
print("metadata records =", len(metadata_records))
print("filesystem images =", len(all_images))

assert len(all_images) == len(metadata_records), (
    f"Metadata/image count mismatch: "
    f"metadata_records={len(metadata_records)}, "
    f"filesystem_images={len(all_images)}"
)

print("metadata/filesystem count check = PASS")

assert sum(
    len(v)
    for v in domain_species[
        "lab"
    ].values()
) == 23147

assert sum(
    len(v)
    for v in domain_species[
        "field"
    ].values()
) == 7719


print()
print(
    "LEAFSNAP FILE COUNTS: PASS"
)

print(
    "LEAFSNAP PROTOCOL AUDIT: PASS"
)
