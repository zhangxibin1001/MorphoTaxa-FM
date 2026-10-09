#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from morphotaxa.config import load_experiment
from morphotaxa.data.manifest_dataset import (
    ManifestDataset,
    discover_manifest_classes,
)
from morphotaxa.models.bioclip25 import load_bioclip25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--model", default="configs/models/bioclip25.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--experiment",
        default="configs/experiments/c0_frozen_cached.yaml",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_experiment(
        args.base,
        args.model,
        args.dataset,
        args.experiment,
    )

    dcfg = cfg["dataset"]
    mcfg = cfg["model"]
    ccfg = cfg.get("cache", {})

    assert ccfg.get("deterministic") is True
    assert str(ccfg.get("preprocess", "")).lower() == "val"

    manifest = Path(dcfg["manifest"])
    root = Path(dcfg["path"])

    class_names, class_to_idx = discover_manifest_classes(
        manifest
    )

    expected = dcfg.get("expected_classes")
    if expected is not None:
        assert len(class_names) == int(expected)

    backbone, _, tf_val = load_bioclip25(
        mcfg.get(
            "open_clip_model",
            "hf-hub:imageomics/bioclip-2.5-vith14",
        ),
        mcfg.get("cache_dir"),
        bool(mcfg.get("offline", True)),
    )

    for p in backbone.parameters():
        p.requires_grad = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    backbone = backbone.to(device)
    backbone.eval()

    batch_size = int(
        ccfg.get("extraction_batch_size", 96)
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest_hash = sha256_file(manifest)

    for split in ("train", "val"):
        print(f"===== CACHE {split.upper()} =====")

        ds = ManifestDataset(
            manifest=manifest,
            split=split,
            class_to_idx=class_to_idx,
            transform=tf_val,   # IMPORTANT: deterministic
            root=root,
        )

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=int(
                cfg["data"].get("workers", 8)
            ),
            pin_memory=bool(
                cfg["data"].get("pin_memory", True)
            ),
        )

        features = []
        targets = []
        sample_ids = []

        with torch.inference_mode():
            for step, (x, y, paths) in enumerate(dl, 1):
                x = x.to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    z = backbone.encode_image(x)

                if isinstance(z, (tuple, list)):
                    z = z[0]

                # Match MorphoTaxa image representation.
                z = F.normalize(
                    z.float(),
                    dim=-1,
                )

                features.append(z.cpu())
                targets.append(y.long().cpu())
                sample_ids.extend(
                    str(p) for p in paths
                )

                if step % 25 == 0:
                    print(
                        f"{split}: "
                        f"{step}/{len(dl)} batches"
                    )

        features = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)

        assert features.ndim == 2
        assert features.shape[0] == targets.shape[0]
        assert features.shape[0] == len(sample_ids)
        assert features.shape[1] == int(
            mcfg.get("embedding_dim", 1024)
        )

        artifact = {
            "split": split,
            "features": features,
            "targets": targets,
            "sample_ids": sample_ids,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "num_classes": len(class_names),
            "feature_dim": features.shape[1],
            "manifest_sha256": manifest_hash,
            "model_id": mcfg.get("open_clip_model"),
            "preprocess": "val",
            "normalized": True,
            "test_evaluated": False,
        }

        path = out / f"{split}.pt"
        torch.save(artifact, path)

        print(
            json.dumps(
                {
                    "split": split,
                    "samples": len(targets),
                    "feature_shape": list(
                        features.shape
                    ),
                    "classes": len(class_names),
                    "manifest_sha256": manifest_hash,
                    "test_evaluated": False,
                    "saved": str(path),
                },
                indent=2,
            )
        )

    print("C0 FEATURE CACHE: PASS")


if __name__ == "__main__":
    main()
