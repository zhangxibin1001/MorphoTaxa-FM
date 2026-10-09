from pathlib import Path

import torch

from morphotaxa.data.manifest_dataset import (
    ManifestDataset,
    discover_manifest_classes,
)
from morphotaxa.models.bioclip25 import (
    load_bioclip25,
)
from morphotaxa.models.modeling import (
    MorphoTaxaModel,
)


MANIFEST = Path(
    "data_protocols/plantnet300k_v2/manifest.csv"
)

ROOT = Path(
    "/root/autodl-tmp/datasets_raw/plantnet300kV2"
)

MODEL_NAME = (
    "local-dir:/root/autodl-tmp/pretrained/huggingface/hub/"
    "models--imageomics--bioclip-2.5-vith14/"
    "snapshots/6e3d04e3d6522012c88181085c5ae666e14c45cd"
)


def main():

    print("===== PlantNet V2 C0 PREFLIGHT =====")

    class_names, class_to_idx = (
        discover_manifest_classes(
            MANIFEST
        )
    )

    print("classes =", len(class_names))
    print("class 0 =", class_names[0])
    print("class 999 =", class_names[-1])

    assert len(class_names) == 1000

    assert (
        class_names[0]
        == "Lactuca virosa L."
    )

    assert (
        class_names[-1]
        == "Stenanona costaricensis R.E.Fr."
    )

    backbone, _, tf_val = load_bioclip25(
        model_name=MODEL_NAME,
        cache_dir=None,
        offline=True,
    )

    for p in backbone.parameters():
        p.requires_grad = False

    model = MorphoTaxaModel(
        backbone=backbone,
        feature_dim=1024,
        num_classes=1000,
        semantic_head=None,
    )

    assert (
        model.classifier.in_features
        == 1024
    )

    assert (
        model.classifier.out_features
        == 1000
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("classifier =", model.classifier)
    print("total params =", total)
    print("trainable params =", trainable)

    EXPECTED_TRAINABLE = 1_025_000
    EXPECTED_TOTAL = 987_134_441

    assert trainable == EXPECTED_TRAINABLE, (
        trainable,
        EXPECTED_TRAINABLE,
    )

    assert total == EXPECTED_TOTAL, (
        total,
        EXPECTED_TOTAL,
    )

    ratio = (
        100.0
        * trainable
        / total
    )

    print(
        "trainable ratio (%) =",
        ratio,
    )

    ds = ManifestDataset(
        manifest=MANIFEST,
        split="val",
        class_to_idx=class_to_idx,
        transform=tf_val,
        root=ROOT,
    )

    print("val samples =", len(ds))

    assert len(ds) == 31115

    sample = ds[0]

    x = sample[0]
    y = sample[1]

    print(
        "sample image shape =",
        tuple(x.shape),
    )

    print(
        "sample target =",
        int(y),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    x = x.unsqueeze(0).to(device)

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits, z = model(x)
        else:
            logits, z = model(x)

    print(
        "embedding shape =",
        tuple(z.shape),
    )

    print(
        "logits shape =",
        tuple(logits.shape),
    )

    print(
        "finite logits =",
        bool(
            torch.isfinite(
                logits
            ).all()
        ),
    )

    assert z.shape == (1, 1024)
    assert logits.shape == (1, 1000)

    assert torch.isfinite(
        logits
    ).all()

    print()
    print(
        "PLANTNET CLASS ORDER: PASS"
    )
    print(
        "PLANTNET C0 PARAM COUNT: PASS"
    )
    print(
        "PLANTNET DATA FORWARD: PASS"
    )
    print(
        "PLANTNET C0 PREFLIGHT: PASS"
    )


if __name__ == "__main__":
    main()
