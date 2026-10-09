from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from morphotaxa.data.manifest_dataset import (
    ManifestDataset,
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

DATA_ROOT = Path(
    "/root/autodl-tmp/datasets_raw/tcmp300_raw"
)

C1_CKPT = Path(
    "outputs/paper/tcmp300_canonical_2026/"
    "c1_lora/seed_42/best.pth"
)

C4A_CKPT = Path(
    "outputs/paper/tcmp300_canonical_2026/"
    "c4a_bsprc/seed_42/best.pth"
)

OUT = Path(
    "outputs/paper_freeze/tcmp300_c4a_seed42/"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def build_stage(
    checkpoint: Path,
    class_names: list[str],
    use_semantic: bool,
):
    backbone, _, tf_val = load_bioclip25(
        MODEL_NAME,
        None,
        True,
    )

    for p in backbone.parameters():
        p.requires_grad = False

    inject_lora_last_blocks(
        backbone,
        last_n_blocks=8,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        targets=("qkv", "proj"),
    )

    semantic = None

    if use_semantic:
        semantic = BoundedSemanticResidual(
            num_classes=len(class_names),
            alpha_max=0.2,
            kappa=1.0,
        )

    model = MorphoTaxaModel(
        backbone=backbone,
        feature_dim=1024,
        num_classes=len(class_names),
        semantic_head=semantic,
    )

    report = load_init_checkpoint_compatible(
        model,
        checkpoint,
    )

    print(
        checkpoint.name,
        "epoch=",
        report["checkpoint_epoch"],
        "missing=",
        report["expected_missing"],
    )

    model = model.to(DEVICE).eval()

    if use_semantic:
        proto = build_text_prototypes(
            model.backbone,
            MODEL_NAME,
            class_names,
            DEVICE,
        )

        model.set_class_prototypes(proto)

    return model, tf_val


@torch.inference_mode()
def evaluate(model, loader):
    logits_all = []
    y_all = []
    paths_all = []

    for x, y, paths in loader:
        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        if DEVICE.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits, _ = model(x)
        else:
            logits, _ = model(x)

        logits_all.append(
            logits.float().cpu().numpy()
        )

        y_all.append(
            y.numpy()
        )

        paths_all.extend(paths)

    return (
        np.concatenate(logits_all),
        np.concatenate(y_all),
        np.asarray(paths_all),
    )


def per_class(y, logits):
    pred = logits.argmax(1)

    p, r, f1, support = (
        precision_recall_fscore_support(
            y,
            pred,
            labels=np.arange(300),
            zero_division=0,
        )
    )

    return pred, p, r, f1, support


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_names, class_to_idx = (
        discover_manifest_classes(
            MANIFEST
        )
    )

    assert len(class_names) == 300

    # Build C1 first to get the exact validation transform.
    c1, tf_val = build_stage(
        C1_CKPT,
        class_names,
        use_semantic=False,
    )

    ds = ManifestDataset(
        manifest=MANIFEST,
        split="val",
        class_to_idx=class_to_idx,
        transform=tf_val,
        root=DATA_ROOT,
    )

    loader = DataLoader(
        ds,
        batch_size=96,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    print("val samples =", len(ds))

    c1_logits, y1, paths1 = evaluate(
        c1,
        loader,
    )

    np.savez_compressed(
        OUT / "c1_val_outputs.npz",
        logits=c1_logits,
        targets=y1,
        paths=paths1,
    )

    del c1
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # C4-A.
    c4a, _ = build_stage(
        C4A_CKPT,
        class_names,
        use_semantic=True,
    )

    c4_logits, y2, paths2 = evaluate(
        c4a,
        loader,
    )

    np.savez_compressed(
        OUT / "c4a_val_outputs.npz",
        logits=c4_logits,
        targets=y2,
        paths=paths2,
    )

    assert np.array_equal(y1, y2)
    assert np.array_equal(paths1, paths2)

    pred1, p1, r1, f1_1, support = (
        per_class(y1, c1_logits)
    )

    pred4, p4, r4, f1_4, support4 = (
        per_class(y2, c4_logits)
    )

    assert np.array_equal(
        support,
        support4,
    )

    # Load learned class-wise alpha.
    ckpt = torch.load(
        C4A_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    raw_alpha = (
        ckpt["model"][
            "semantic_head.raw_alpha"
        ]
        .float()
    )

    alpha = (
        0.2
        * torch.tanh(raw_alpha)
    ).numpy()

    rows = []

    for c in range(300):
        rows.append({
            "label": c,
            "class_name": class_names[c],
            "semantic_name":
                clean_semantic_class_name(
                    class_names[c]
                ),
            "support": int(support[c]),
            "alpha": float(alpha[c]),
            "abs_alpha": float(
                abs(alpha[c])
            ),
            "c1_precision":
                float(p1[c]),
            "c1_recall":
                float(r1[c]),
            "c1_f1":
                float(f1_1[c]),
            "c4a_precision":
                float(p4[c]),
            "c4a_recall":
                float(r4[c]),
            "c4a_f1":
                float(f1_4[c]),
            "delta_recall":
                float(r4[c] - r1[c]),
            "delta_f1":
                float(f1_4[c] - f1_1[c]),
        })

    csv_path = (
        OUT
        / "c1_vs_c4a_classwise.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    c1_correct = pred1 == y1
    c4_correct = pred4 == y2

    fixed = (
        (~c1_correct)
        & c4_correct
    )

    broken = (
        c1_correct
        & (~c4_correct)
    )

    changed = pred1 != pred4

    print()
    print("===== SAMPLE-LEVEL CHANGE =====")
    print(
        "C1 correct =",
        int(c1_correct.sum()),
    )
    print(
        "C4A correct =",
        int(c4_correct.sum()),
    )
    print(
        "wrong -> correct =",
        int(fixed.sum()),
    )
    print(
        "correct -> wrong =",
        int(broken.sum()),
    )
    print(
        "prediction changed =",
        int(changed.sum()),
    )

    print()
    print("===== CLASS-LEVEL CHANGE =====")

    improved = np.array(
        [r["delta_f1"] for r in rows]
    ) > 0

    degraded = np.array(
        [r["delta_f1"] for r in rows]
    ) < 0

    unchanged = ~(
        improved | degraded
    )

    print(
        "improved classes =",
        int(improved.sum()),
    )
    print(
        "degraded classes =",
        int(degraded.sum()),
    )
    print(
        "unchanged classes =",
        int(unchanged.sum()),
    )

    # Bottom quartile of C1 classes.
    c1_f1_array = np.asarray(
        [r["c1_f1"] for r in rows]
    )

    delta_f1 = np.asarray(
        [r["delta_f1"] for r in rows]
    )

    q25 = np.quantile(
        c1_f1_array,
        0.25,
    )

    q75 = np.quantile(
        c1_f1_array,
        0.75,
    )

    hard = c1_f1_array <= q25
    easy = c1_f1_array >= q75

    print()
    print("===== HARD VS EASY =====")
    print("C1 F1 q25 =", float(q25))
    print("C1 F1 q75 =", float(q75))
    print(
        "hard-class mean delta F1 =",
        float(delta_f1[hard].mean()),
    )
    print(
        "easy-class mean delta F1 =",
        float(delta_f1[easy].mean()),
    )

    try:
        from scipy.stats import (
            spearmanr,
        )

        abs_alpha = np.abs(alpha)

        rho1, pval1 = spearmanr(
            abs_alpha,
            np.abs(delta_f1),
        )

        rho2, pval2 = spearmanr(
            alpha,
            delta_f1,
        )

        print()
        print("===== CORRELATION =====")
        print(
            "|alpha| vs |delta F1|:",
            "rho=",
            float(rho1),
            "p=",
            float(pval1),
        )

        print(
            "alpha vs delta F1:",
            "rho=",
            float(rho2),
            "p=",
            float(pval2),
        )

    except Exception as e:
        print(
            "Spearman skipped:",
            repr(e),
        )

    print()
    print("===== TOP 15 IMPROVED =====")

    for r in sorted(
        rows,
        key=lambda x: x["delta_f1"],
        reverse=True,
    )[:15]:
        print(
            f'{r["label"]:3d}',
            r["semantic_name"],
            f'alpha={r["alpha"]:+.4f}',
            f'C1={r["c1_f1"]:.4f}',
            f'C4A={r["c4a_f1"]:.4f}',
            f'delta={r["delta_f1"]:+.4f}',
        )

    print()
    print("===== TOP 15 DEGRADED =====")

    for r in sorted(
        rows,
        key=lambda x: x["delta_f1"],
    )[:15]:
        print(
            f'{r["label"]:3d}',
            r["semantic_name"],
            f'alpha={r["alpha"]:+.4f}',
            f'C1={r["c1_f1"]:.4f}',
            f'C4A={r["c4a_f1"]:.4f}',
            f'delta={r["delta_f1"]:+.4f}',
        )

    print()
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
