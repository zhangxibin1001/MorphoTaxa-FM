from __future__ import annotations

import os
from pathlib import Path

import torch

from morphotaxa.data.manifest_dataset import (
    discover_manifest_classes,
)
from morphotaxa.engine.checkpoint_compat import (
    load_init_checkpoint_compatible,
)
from morphotaxa.models.bioclip25 import (
    load_bioclip25,
)
from morphotaxa.models.lora import (
    inject_lora_last_blocks,
)
from morphotaxa.models.c4a import (
    BoundedSemanticResidual,
)
from morphotaxa.models.modeling import (
    MorphoTaxaModel,
)
from morphotaxa.models.prototypes import (
    build_text_prototypes,
    clean_semantic_class_name,
)


MODEL_NAME = (
    "local-dir:/root/autodl-tmp/pretrained/huggingface/hub/"
    "models--imageomics--bioclip-2.5-vith14/"
    "snapshots/6e3d04e3d6522012c88181085c5ae666e14c45cd"
)

MANIFEST = Path(
    "data_protocols/tcmp300/manifest.csv"
)

C1_CKPT = Path(
    os.environ.get(
        "C1_CKPT",
        "outputs/paper/tcmp300_canonical_2026/"
        "c1_lora/seed_42/best.pth",
    )
)


def main():
    print("===== C4-A PREFLIGHT =====")

    class_names, class_to_idx = (
        discover_manifest_classes(
            MANIFEST
        )
    )

    assert len(class_names) == 300
    assert len(class_to_idx) == 300

    print("classes =", len(class_names))

    print("first semantic names:")
    for name in class_names[:3]:
        print(
            " ",
            name,
            "->",
            clean_semantic_class_name(name),
        )

    backbone, _, _ = load_bioclip25(
        MODEL_NAME,
        None,
        offline=True,
    )

    for p in backbone.parameters():
        p.requires_grad = False

    replaced = inject_lora_last_blocks(
        backbone,
        last_n_blocks=8,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        targets=("qkv", "proj"),
    )

    assert len(replaced) == 16

    semantic = BoundedSemanticResidual(
        num_classes=300,
        alpha_max=0.2,
        kappa=1.0,
    )

    model = MorphoTaxaModel(
        backbone=backbone,
        feature_dim=1024,
        num_classes=300,
        semantic_head=semantic,
    )

    report = load_init_checkpoint_compatible(
        model,
        C1_CKPT,
    )

    print("checkpoint =", report["checkpoint"])
    print(
        "checkpoint_epoch =",
        report["checkpoint_epoch"],
    )
    print(
        "remapped_count =",
        report["remapped_count"],
    )
    print(
        "expected_missing =",
        report["expected_missing"],
    )

    assert report["checkpoint_epoch"] == 10

    # C1 already contains the parametrized LoRA structure,
    # therefore no C0->C1 remapping should be required.
    assert report["remapped_count"] == 0

    # C4-A introduces only raw_alpha.
    assert report["expected_missing"] == [
        "semantic_head.raw_alpha"
    ]

    # Reproduce trainer's C4-A freeze policy.
    for p in model.parameters():
        p.requires_grad = False

    for p in model.semantic_head.parameters():
        p.requires_grad = True

    trainable = [
        (n, p.numel())
        for n, p in model.named_parameters()
        if p.requires_grad
    ]

    print("trainable =", trainable)

    assert trainable == [
        ("semantic_head.raw_alpha", 300)
    ]

    assert torch.count_nonzero(
        model.semantic_head.raw_alpha
    ).item() == 0

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device).eval()

    prototypes = build_text_prototypes(
        model.backbone,
        MODEL_NAME,
        class_names,
        device,
    )

    print(
        "prototype shape =",
        tuple(prototypes.shape),
    )

    assert prototypes.shape == (300, 1024)
    assert torch.isfinite(prototypes).all()

    norms = prototypes.float().norm(
        dim=-1
    )

    print(
        "prototype norm min/max =",
        float(norms.min()),
        float(norms.max()),
    )

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    )

    model.set_class_prototypes(
        prototypes
    )

    # Real model-level identity test.
    torch.manual_seed(42)

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        z = model.encode_image(x)

        base_logits = model.classifier(z)

        c4a_logits = model.semantic_head(
            base_logits,
            z,
            model.class_prototypes,
        )

    max_diff = float(
        (base_logits - c4a_logits)
        .abs()
        .max()
        .cpu()
    )

    print(
        "epoch0 logit max diff =",
        max_diff,
    )

    print(
        "alpha abs max =",
        float(
            model.semantic_head
            .bounded_alpha
            .abs()
            .max()
            .cpu()
        ),
    )

    assert max_diff == 0.0

    print()
    print(
        "C1 -> C4A CHECKPOINT: PASS"
    )
    print(
        "C4A TRAINABLE SCOPE: PASS"
    )
    print(
        "TEXT PROTOTYPES: PASS"
    )
    print(
        "C4A EPOCH-0 IDENTITY: PASS"
    )


if __name__ == "__main__":
    main()
