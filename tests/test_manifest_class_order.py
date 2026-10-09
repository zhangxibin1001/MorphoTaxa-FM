import csv

from morphotaxa.data.manifest_dataset import (
    discover_manifest_classes,
)


def test_manifest_class_order_follows_label_not_name(
    tmp_path,
):
    p = tmp_path / "manifest.csv"

    rows = [
        {
            "path": "a.jpg",
            "split": "train",
            "label": 0,
            "class_name": "Lactuca virosa L.",
        },
        {
            "path": "b.jpg",
            "split": "train",
            "label": 1,
            "class_name": "Abeliophyllum distichum Nakai",
        },
    ]

    with p.open(
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
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    names, class_to_idx = (
        discover_manifest_classes(p)
    )

    # Alphabetical sorting would incorrectly
    # put Abeliophyllum first.
    assert names == [
        "Lactuca virosa L.",
        "Abeliophyllum distichum Nakai",
    ]

    assert (
        class_to_idx[
            "Lactuca virosa L."
        ]
        == 0
    )

    assert (
        class_to_idx[
            "Abeliophyllum distichum Nakai"
        ]
        == 1
    )
