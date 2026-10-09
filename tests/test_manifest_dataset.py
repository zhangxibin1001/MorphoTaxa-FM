from pathlib import Path

from PIL import Image

from morphotaxa.data.manifest_dataset import (
    ManifestDataset,
    discover_manifest_classes,
)


def test_manifest_dataset(tmp_path):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"

    Image.new(
        "RGB",
        (16, 16),
    ).save(img1)

    Image.new(
        "RGB",
        (16, 16),
    ).save(img2)

    manifest = tmp_path / "manifest.csv"

    manifest.write_text(
        "path,class_name,split\n"
        f"{img1},class_a,train\n"
        f"{img2},class_a,val\n",
        encoding="utf-8",
    )

    classes, mapping = (
        discover_manifest_classes(
            manifest
        )
    )

    assert classes == ["class_a"]
    assert mapping == {"class_a": 0}

    ds = ManifestDataset(
        manifest,
        split="train",
        class_to_idx=mapping,
    )

    assert len(ds) == 1

    image, target, path = ds[0]

    assert image.size == (16, 16)
    assert target == 0
    assert path == str(img1)
