#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from morphotaxa.config import load_experiment
from morphotaxa.reproducibility import seed_everything
from morphotaxa.metrics import classification_metrics
from morphotaxa.calibration import (
    expected_calibration_error,
    multiclass_brier,
    nll,
)
from morphotaxa.models.modeling import build_model


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_cache(path: Path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def validate_caches(tr, va):
    assert tr["split"] == "train"
    assert va["split"] == "val"

    assert tr["test_evaluated"] is False
    assert va["test_evaluated"] is False

    assert tr["preprocess"] == "val"
    assert va["preprocess"] == "val"

    assert tr["normalized"] is True
    assert va["normalized"] is True

    assert tr["manifest_sha256"] == va["manifest_sha256"]
    assert tr["class_names"] == va["class_names"]
    assert tr["class_to_idx"] == va["class_to_idx"]

    assert tr["num_classes"] == va["num_classes"]
    assert tr["feature_dim"] == va["feature_dim"]

    assert tr["features"].ndim == 2
    assert va["features"].ndim == 2

    assert tr["features"].shape[0] == tr["targets"].shape[0]
    assert va["features"].shape[0] == va["targets"].shape[0]


def evaluate(
    head,
    features,
    targets,
    batch_size,
    device,
):
    head.eval()

    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    logits_all = []
    targets_all = []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            logits = head(x)

            logits_all.append(
                logits.float().cpu()
            )
            targets_all.append(
                y.long().cpu()
            )

    logits = torch.cat(
        logits_all,
        dim=0,
    ).numpy()

    y = torch.cat(
        targets_all,
        dim=0,
    ).numpy()

    metrics = classification_metrics(
        logits,
        y,
    )

    metrics.update(
        {
            "ece": expected_calibration_error(
                logits,
                y,
            ),
            "nll": nll(
                logits,
                y,
            ),
            "brier": multiclass_brier(
                logits,
                y,
            ),
        }
    )

    return metrics, logits, y


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Formal frozen-feature C0 linear probe. "
            "Uses only cached train/val BioCLIP2.5 features."
        )
    )

    ap.add_argument(
        "--base",
        default="configs/base.yaml",
    )
    ap.add_argument(
        "--model",
        default="configs/models/bioclip25.yaml",
    )
    ap.add_argument(
        "--dataset",
        required=True,
    )
    ap.add_argument(
        "--experiment",
        default=(
            "configs/experiments/"
            "c0_frozen_cached.yaml"
        ),
    )
    ap.add_argument(
        "--cache-dir",
        required=True,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    cfg = load_experiment(
        args.base,
        args.model,
        args.dataset,
        args.experiment,
    )

    cfg.setdefault(
        "project",
        {},
    )["seed"] = args.seed

    seed_everything(
        args.seed
    )

    tcfg = cfg["train"]

    # Frozen protocol assertions.
    assert int(tcfg["epochs"]) == 50
    assert int(tcfg["batch_size"]) == 1024
    assert float(tcfg["lr"]) == 1e-3
    assert float(tcfg["weight_decay"]) == 0.01
    assert float(
        tcfg.get(
            "label_smoothing",
            0.0,
        )
    ) == 0.0

    cache_dir = Path(
        args.cache_dir
    )

    tr = load_cache(
        cache_dir / "train.pt"
    )
    va = load_cache(
        cache_dir / "val.pt"
    )

    validate_caches(
        tr,
        va,
    )

    expected_classes = cfg[
        "dataset"
    ].get(
        "expected_classes"
    )

    if expected_classes is not None:
        assert (
            tr["num_classes"]
            == int(expected_classes)
        )

    num_classes = int(
        tr["num_classes"]
    )
    feature_dim = int(
        tr["feature_dim"]
    )

    train_features = (
        tr["features"]
        .float()
        .contiguous()
    )
    train_targets = (
        tr["targets"]
        .long()
        .contiguous()
    )

    val_features = (
        va["features"]
        .float()
        .contiguous()
    )
    val_targets = (
        va["targets"]
        .long()
        .contiguous()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    head = nn.Linear(
        feature_dim,
        num_classes,
    ).to(device)

    trainable = sum(
        p.numel()
        for p in head.parameters()
        if p.requires_grad
    )

    expected_trainable = (
        feature_dim * num_classes
        + num_classes
    )

    assert (
        trainable
        == expected_trainable
    )

    print(
        "===== FORMAL C0 ====="
    )
    print(
        "device =",
        device,
    )
    print(
        "train samples =",
        len(train_targets),
    )
    print(
        "val samples =",
        len(val_targets),
    )
    print(
        "classes =",
        num_classes,
    )
    print(
        "feature_dim =",
        feature_dim,
    )
    print(
        "trainable params =",
        trainable,
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(
            tcfg["weight_decay"]
        ),
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(
            tcfg.get(
                "label_smoothing",
                0.0,
            )
        )
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        TensorDataset(
            train_features,
            train_targets,
        ),
        batch_size=int(
            tcfg["batch_size"]
        ),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    out = Path(
        args.output
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_macro_f1 = -1.0
    best_epoch = None
    best_state = None
    best_metrics = None
    best_logits = None
    best_targets = None

    epochs = int(
        tcfg["epochs"]
    )

    start_all = time.time()

    for epoch in range(
        1,
        epochs + 1,
    ):
        start = time.time()

        head.train()

        total_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(
                device,
                non_blocking=True,
            )
            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = head(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.detach())
                * len(y)
            )
            seen += len(y)

        metrics, logits_np, y_np = evaluate(
            head,
            val_features,
            val_targets,
            batch_size=int(
                tcfg.get(
                    "eval_batch_size",
                    2048,
                )
            ),
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": (
                total_loss
                / max(1, seen)
            ),
            "epoch_seconds": (
                time.time()
                - start
            ),
            **{
                f"val_{k}": v
                for k, v
                in metrics.items()
            },
        }

        history.append(
            row
        )

        dump_json(
            history,
            out / "history.json",
        )

        print(
            json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

        if (
            metrics["macro_f1"]
            > best_macro_f1
        ):
            best_macro_f1 = float(
                metrics["macro_f1"]
            )
            best_epoch = epoch

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in head.state_dict().items()
            }

            best_metrics = copy.deepcopy(
                metrics
            )

            best_logits = (
                logits_np.copy()
            )
            best_targets = (
                y_np.copy()
            )

            torch.save(
                {
                    "model": best_state,
                    "epoch": best_epoch,
                    "val": best_metrics,
                    "feature_dim": feature_dim,
                    "num_classes": num_classes,
                    "class_names": tr[
                        "class_names"
                    ],
                    "manifest_sha256": tr[
                        "manifest_sha256"
                    ],
                    "seed": args.seed,
                    "test_evaluated": False,
                },
                out
                / "classifier_best.pth",
            )

    assert best_state is not None
    assert best_metrics is not None
    assert best_epoch is not None

    np.save(
        out / "val_logits.npy",
        best_logits,
    )
    np.save(
        out / "val_targets.npy",
        best_targets,
    )

    (
        out
        / "val_sample_ids.txt"
    ).write_text(
        "\n".join(
            str(x)
            for x in va[
                "sample_ids"
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dump_json(
        best_metrics,
        out / "val_metrics.json",
    )

    # --------------------------------------------------
    # Build a FULL C0 checkpoint only once, after the
    # best cached-feature classifier has been selected.
    # This keeps training truly cached-feature-only,
    # while preserving compatibility with existing C1.
    # --------------------------------------------------

    print(
        "\nBuilding full C0 "
        "checkpoint for C1 compatibility..."
    )

    full_model = build_model(
        cfg,
        num_classes,
    )

    full_model.classifier.load_state_dict(
        best_state
    )

    full_model = full_model.cpu()

    total_params = sum(
        p.numel()
        for p in full_model.parameters()
    )

    full_trainable = sum(
        p.numel()
        for p in full_model.parameters()
        if p.requires_grad
    )

    assert (
        full_trainable
        == expected_trainable
    )

    checkpoint = {
        "model": full_model.state_dict(),
        "cfg": cfg,
        "epoch": best_epoch,
        "val": best_metrics,
        "cache_provenance": {
            "manifest_sha256": tr[
                "manifest_sha256"
            ],
            "class_names": tr[
                "class_names"
            ],
            "class_to_idx": tr[
                "class_to_idx"
            ],
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "preprocess": "val",
            "normalized": True,
            "test_evaluated": False,
        },
    }

    torch.save(
        checkpoint,
        out / "best.pth",
    )

    summary = {
        "status": "complete",
        "dataset": cfg[
            "dataset"
        ]["name"],
        "method": (
            "c0_frozen_cached"
        ),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": (
            best_macro_f1
        ),
        "metrics": best_metrics,
        "parameter_counts": {
            "total": total_params,
            "trainable": (
                full_trainable
            ),
        },
        "cache": {
            "train_samples": len(
                train_targets
            ),
            "val_samples": len(
                val_targets
            ),
            "feature_dim": feature_dim,
            "num_classes": (
                num_classes
            ),
            "manifest_sha256": tr[
                "manifest_sha256"
            ],
            "preprocess": "val",
            "normalized": True,
        },
        "training_seconds": (
            time.time()
            - start_all
        ),
        "test_evaluated": False,
    }

    dump_json(
        summary,
        out / "summary.json",
    )

    print(
        "\n===== C0 COMPLETE ====="
    )
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nFORMAL C0 FROZEN-CACHE: PASS"
    )


if __name__ == "__main__":
    main()
