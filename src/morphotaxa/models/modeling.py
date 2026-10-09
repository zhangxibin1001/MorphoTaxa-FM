from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .bioclip25 import load_bioclip25
from .lora import inject_lora_last_blocks
from .c4a import BoundedSemanticResidual


class MorphoTaxaModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, semantic_head: BoundedSemanticResidual | None = None):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.semantic_head = semantic_head
        self.register_buffer("class_prototypes", torch.empty(0), persistent=False)

    def set_class_prototypes(self, prototypes: torch.Tensor):
        if prototypes.ndim != 2 or prototypes.shape[0] != self.classifier.out_features:
            raise ValueError("prototype shape must be [num_classes, feature_dim]")
        self.class_prototypes = prototypes.detach().clone()

    def encode_image(self, x):
        if hasattr(self.backbone, "encode_image"):
            z = self.backbone.encode_image(x)
        elif hasattr(self.backbone, "forward_features"):
            z = self.backbone.forward_features(x)
            if z.ndim == 3:
                z = z[:, 0]
        else:
            z = self.backbone(x)
        if isinstance(z, (tuple, list)):
            z = z[0]
        return F.normalize(z, dim=-1)

    def forward(self, x):
        z = self.encode_image(x)
        logits = self.classifier(z)
        if self.semantic_head is not None:
            if self.class_prototypes.numel() == 0:
                raise RuntimeError("C4-A requires class prototypes")
            logits = self.semantic_head(logits, z, self.class_prototypes)
        return logits, z


def build_model(cfg, num_classes: int):
    mcfg = cfg["model"]
    backbone, _, _ = load_bioclip25(
        model_name=mcfg.get("open_clip_model", "hf-hub:imageomics/bioclip-2.5-vith14"),
        cache_dir=mcfg.get("cache_dir"),
        offline=bool(mcfg.get("offline", True)),
    )
    for p in backbone.parameters():
        p.requires_grad = False
    lcfg = mcfg.get("lora", {})
    if lcfg.get("enabled", False):
        inject_lora_last_blocks(
            backbone,
            last_n_blocks=lcfg.get("last_n_blocks", 8),
            rank=lcfg.get("rank", 8),
            alpha=lcfg.get("alpha", 16.0),
            dropout=lcfg.get("dropout", 0.0),
            targets=tuple(lcfg.get("targets", ["qkv", "proj"])),
        )
    sem = None
    scfg = mcfg.get("semantic", {})
    if scfg.get("enabled", False):
        sem = BoundedSemanticResidual(num_classes, scfg.get("alpha_max", 0.2), scfg.get("kappa", 1.0))
    return MorphoTaxaModel(backbone, int(mcfg.get("embedding_dim", 1024)), num_classes, sem)
