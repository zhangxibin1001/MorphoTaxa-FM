from pathlib import Path
from collections import Counter

ROOT_CANDIDATES = [
    Path("/root/autodl-tmp/datasets/leafsnap"),
    Path("/root/autodl-tmp/datasets_raw/leafsnap"),
    Path("/root/autodl-tmp/datasets/LeafSnap"),
    Path("/root/autodl-tmp/datasets_raw/LeafSnap"),
]

EXTS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp",
    ".tif", ".tiff",
}

root = None

for p in ROOT_CANDIDATES:
    if p.exists():
        root = p
        break

if root is None:
    raise RuntimeError(
        "LeafSnap root not found.\n"
        "Checked:\n"
        + "\n".join(
            str(x)
            for x in ROOT_CANDIDATES
        )
    )

print(
    "===== LEAFSNAP RAW AUDIT ====="
)

print(
    "root =",
    root,
)

images = [
    p
    for p in root.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower()
        in EXTS
    )
]

print(
    "images =",
    len(images),
)

assert images, (
    "No images found"
)

print(
    "\n===== TOP-LEVEL ====="
)

for p in sorted(
    root.iterdir()
):
    print(
        p.name,
        "DIR"
        if p.is_dir()
        else "FILE",
    )


print(
    "\n===== POSSIBLE METADATA ====="
)

meta_exts = {
    ".txt",
    ".csv",
    ".tsv",
    ".json",
}

meta_files = [
    p
    for p in root.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower()
        in meta_exts
    )
]

for p in meta_files[:100]:
    print(
        p.relative_to(root)
    )

print(
    "metadata files =",
    len(meta_files),
)


print(
    "\n===== PATH TOKEN COUNTS ====="
)

domain_counts = Counter()

for p in images:
    s = str(
        p.relative_to(root)
    ).lower()

    if "field" in s:
        domain_counts[
            "field"
        ] += 1

    elif "lab" in s:
        domain_counts[
            "lab"
        ] += 1

    else:
        domain_counts[
            "unknown"
        ] += 1

print(
    dict(domain_counts)
)


print(
    "\n===== SAMPLE IMAGE PATHS ====="
)

for p in images[:30]:
    print(
        p.relative_to(root)
    )


print(
    "\n===== DIRECTORY DEPTH ====="
)

depths = Counter(
    len(
        p.relative_to(root).parts
    )
    for p in images
)

print(
    dict(
        sorted(
            depths.items()
        )
    )
)


print()
print(
    "LEAFSNAP RAW AUDIT: PASS"
)
