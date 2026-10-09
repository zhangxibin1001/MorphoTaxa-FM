from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize



class LoRALinear(nn.Module):
    """
    Backward-compatible LoRA wrapper for nn.Linear.

    Kept for the original project API/tests. The BioCLIP/OpenCLIP path
    uses weight parametrization below, because MultiheadAttention stores
    QKV as in_proj_weight rather than an nn.Linear qkv module.

    At initialization B=0, therefore:
        LoRALinear(x) == base(x)
    exactly up to floating-point arithmetic.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        r: int | None = None,
    ):
        super().__init__()

        if not isinstance(base, nn.Linear):
            raise TypeError(
                "LoRALinear expects an nn.Linear base layer"
            )

        if r is not None:
            rank = r

        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.scaling = self.scale

        # Frozen pretrained/base parameters.
        for param in self.base.parameters():
            param.requires_grad = False

        self.dropout = (
            nn.Dropout(float(dropout))
            if float(dropout) > 0.0
            else nn.Identity()
        )

        self.lora_a = nn.Linear(
            self.base.in_features,
            self.rank,
            bias=False,
        )

        self.lora_b = nn.Linear(
            self.rank,
            self.base.out_features,
            bias=False,
        )

        nn.init.kaiming_uniform_(
            self.lora_a.weight,
            a=math.sqrt(5),
        )

        # Identity initialization.
        nn.init.zeros_(self.lora_b.weight)

    @property
    def lora_A(self):
        return self.lora_a

    @property
    def lora_B(self):
        return self.lora_b

    @property
    def A(self):
        return self.lora_a

    @property
    def B(self):
        return self.lora_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        update = self.lora_b(
            self.lora_a(
                self.dropout(x)
            )
        )

        return base_out + self.scale * update


class LoRAWeight(nn.Module):
    """
    Low-rank additive weight parametrization:

        W' = W + (alpha / rank) * B @ A

    B is zero-initialized, therefore injection is an exact identity
    transformation at epoch 0.

    This implementation is especially useful for OpenCLIP attention,
    where QKV may be stored as a raw fused Parameter
    (`in_proj_weight`) rather than an nn.Linear module.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank

        self.lora_a = nn.Parameter(
            torch.empty(self.rank, in_features)
        )

        self.lora_b = nn.Parameter(
            torch.zeros(out_features, self.rank)
        )

        nn.init.kaiming_uniform_(
            self.lora_a,
            a=math.sqrt(5),
        )

        # Critical:
        # exact identity at initialization.
        nn.init.zeros_(self.lora_b)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = torch.matmul(
            self.lora_b,
            self.lora_a,
        )

        delta = delta.to(
            device=weight.device,
            dtype=weight.dtype,
        )

        return weight + self.scale * delta


def _find_visual_blocks(model: nn.Module):
    """
    Find the visual transformer blocks.

    Supported layouts include:

    OpenCLIP:
        model.visual.transformer.resblocks

    timm/OpenCLIP timm tower:
        model.visual.trunk.blocks

    timm:
        model.blocks

    Other common visual layouts:
        model.visual.blocks
        model.trunk.blocks
    """

    candidates = []

    # Native OpenCLIP VisionTransformer
    visual = getattr(model, "visual", None)

    if visual is not None:
        transformer = getattr(
            visual,
            "transformer",
            None,
        )

        if (
            transformer is not None
            and hasattr(transformer, "resblocks")
        ):
            candidates.append(
                (
                    transformer.resblocks,
                    "visual.transformer.resblocks",
                )
            )

        trunk = getattr(visual, "trunk", None)

        if (
            trunk is not None
            and hasattr(trunk, "blocks")
        ):
            candidates.append(
                (
                    trunk.blocks,
                    "visual.trunk.blocks",
                )
            )

        if hasattr(visual, "blocks"):
            candidates.append(
                (
                    visual.blocks,
                    "visual.blocks",
                )
            )

    # Standard timm
    if hasattr(model, "blocks"):
        candidates.append(
            (
                model.blocks,
                "blocks",
            )
        )

    trunk = getattr(model, "trunk", None)

    if (
        trunk is not None
        and hasattr(trunk, "blocks")
    ):
        candidates.append(
            (
                trunk.blocks,
                "trunk.blocks",
            )
        )

    for blocks, prefix in candidates:
        try:
            n = len(blocks)
        except Exception:
            continue

        if n > 0:
            return blocks, prefix

    raise AttributeError(
        "Could not locate visual transformer blocks. "
        "Expected one of: "
        "visual.transformer.resblocks, "
        "visual.trunk.blocks, visual.blocks, "
        "blocks, trunk.blocks."
    )


def _register_weight_lora(
    module: nn.Module,
    weight_name: str,
    rank: int,
    alpha: float,
) -> None:
    """
    Apply LoRA directly to a matrix Parameter.

    The original weight remains frozen; A/B are trainable.
    """

    weight = getattr(module, weight_name, None)

    if weight is None:
        raise AttributeError(
            f"{type(module).__name__}.{weight_name} is None"
        )

    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"{type(module).__name__}.{weight_name} "
            f"is not a tensor"
        )

    if weight.ndim != 2:
        raise ValueError(
            f"{weight_name} must be 2D, "
            f"got shape={tuple(weight.shape)}"
        )

    out_features, in_features = weight.shape

    parametrize.register_parametrization(
        module,
        weight_name,
        LoRAWeight(
            out_features=out_features,
            in_features=in_features,
            rank=rank,
            alpha=alpha,
        ),
    )

    # Explicitly preserve the frozen pretrained weight.
    original = getattr(
        module.parametrizations,
        weight_name,
    ).original

    original.requires_grad = False


def _inject_timm_target(
    attn: nn.Module,
    target: str,
    rank: int,
    alpha: float,
):
    """
    timm attention:
        attn.qkv.weight
        attn.proj.weight
    """

    layer = getattr(attn, target, None)

    if not isinstance(layer, nn.Linear):
        return False

    _register_weight_lora(
        layer,
        "weight",
        rank,
        alpha,
    )

    return True


def _inject_openclip_qkv(
    attn: nn.Module,
    rank: int,
    alpha: float,
) -> list[str]:
    """
    Native OpenCLIP / torch MultiheadAttention.

    Most ViTs use:
        in_proj_weight: [3D, D]

    Some variants expose:
        q_proj_weight
        k_proj_weight
        v_proj_weight
    """

    names: list[str] = []

    fused = getattr(
        attn,
        "in_proj_weight",
        None,
    )

    if isinstance(fused, torch.Tensor):
        _register_weight_lora(
            attn,
            "in_proj_weight",
            rank,
            alpha,
        )

        names.append("in_proj_weight")
        return names

    for name in (
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
    ):
        weight = getattr(attn, name, None)

        if isinstance(weight, torch.Tensor):
            _register_weight_lora(
                attn,
                name,
                rank,
                alpha,
            )
            names.append(name)

    return names


def _inject_openclip_proj(
    attn: nn.Module,
    rank: int,
    alpha: float,
) -> bool:
    """
    Native OpenCLIP / MultiheadAttention output projection:
        attn.out_proj.weight
    """

    out_proj = getattr(
        attn,
        "out_proj",
        None,
    )

    if not isinstance(out_proj, nn.Linear):
        return False

    _register_weight_lora(
        out_proj,
        "weight",
        rank,
        alpha,
    )

    return True


def inject_lora_last_blocks(
    model: nn.Module,
    last_n_blocks: int = 8,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Iterable[str] = ("qkv", "proj"),
) -> list[str]:
    """
    Inject selective attention LoRA into the last N visual blocks.

    Logical paper-level targets:
        qkv
        proj

    Mapping:

    timm:
        qkv  -> attn.qkv.weight
        proj -> attn.proj.weight

    OpenCLIP:
        qkv  -> attn.in_proj_weight
                or q/k/v_proj_weight
        proj -> attn.out_proj.weight

    Returns audited names of injected parameters.
    """

    # Weight-parametrized LoRA is mathematically equivalent to
    # normal LoRA when dropout=0. The paper/default configuration
    # intentionally uses dropout=0.
    if float(dropout) != 0.0:
        raise ValueError(
            "This OpenCLIP-compatible LoRA implementation "
            "currently requires dropout=0.0. "
            f"Received dropout={dropout}."
        )

    blocks, prefix = _find_visual_blocks(model)

    n_blocks = len(blocks)

    if n_blocks <= 0:
        raise RuntimeError(
            "Visual transformer contains zero blocks"
        )

    last_n_blocks = int(last_n_blocks)

    if last_n_blocks <= 0:
        raise ValueError(
            "last_n_blocks must be > 0"
        )

    start = max(
        0,
        n_blocks - last_n_blocks,
    )

    targets = tuple(
        str(x).lower()
        for x in targets
    )

    replaced: list[str] = []

    for bi in range(start, n_blocks):
        block = blocks[bi]

        attn = getattr(
            block,
            "attn",
            None,
        )

        if attn is None:
            continue

        if "qkv" in targets:
            # timm-style first
            if _inject_timm_target(
                attn,
                "qkv",
                rank,
                alpha,
            ):
                replaced.append(
                    f"{prefix}.{bi}.attn.qkv.weight"
                )

            else:
                names = _inject_openclip_qkv(
                    attn,
                    rank,
                    alpha,
                )

                for name in names:
                    replaced.append(
                        f"{prefix}.{bi}.attn.{name}"
                    )

        if "proj" in targets:
            # timm-style
            if _inject_timm_target(
                attn,
                "proj",
                rank,
                alpha,
            ):
                replaced.append(
                    f"{prefix}.{bi}.attn.proj.weight"
                )

            # native OpenCLIP
            elif _inject_openclip_proj(
                attn,
                rank,
                alpha,
            ):
                replaced.append(
                    f"{prefix}.{bi}.attn.out_proj.weight"
                )

    if not replaced:
        raise RuntimeError(
            "No LoRA targets were injected. "
            f"block_path={prefix}, "
            f"n_blocks={n_blocks}, "
            f"targets={targets}. "
            "Inspect attention module structure."
        )

    return replaced
