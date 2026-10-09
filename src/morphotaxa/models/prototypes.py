from __future__ import annotations

import re

import torch
import torch.nn.functional as F


def clean_semantic_class_name(name: str) -> str:
    """
    Remove dataset-internal numeric class prefixes while preserving
    the biological/scientific name.

    Examples:
        001.Veronica persica Poir. -> Veronica persica Poir.
        002_Lysimachia_clethroides -> Lysimachia clethroides

    This transformation changes only the text prompt, never the
    classifier class ordering.
    """
    name = str(name).strip()

    # Remove zero-padded dataset index prefixes such as:
    # 001. / 001_ / 001-
    name = re.sub(
        r"^\s*\d+\s*[\.\-_]\s*",
        "",
        name,
    )

    name = name.replace("_", " ")

    return " ".join(name.split())


DEFAULT_TEMPLATES = (
    "a photo of {name}",
    "a botanical photograph of {name}",
    "a leaf of {name}",
    "a medicinal plant {name}",
)


def build_text_prototypes(model, model_name: str, class_names: list[str], device, templates=DEFAULT_TEMPLATES) -> torch.Tensor:
    try:
        import open_clip
    except ImportError as e:
        raise RuntimeError("open_clip is required to create text prototypes") from e
    if not hasattr(model, "encode_text"):
        raise RuntimeError("Selected backbone does not expose encode_text")
    tokenizer = open_clip.get_tokenizer(model_name)
    out = []
    model.eval()
    with torch.no_grad():
        for name in class_names:
            semantic_name = clean_semantic_class_name(name)
            texts = [
                t.format(name=semantic_name)
                for t in templates
            ]
            tokens = tokenizer(texts).to(device)
            z = model.encode_text(tokens)
            z = F.normalize(z, dim=-1).mean(0)
            out.append(F.normalize(z, dim=-1))
    return torch.stack(out, 0)
