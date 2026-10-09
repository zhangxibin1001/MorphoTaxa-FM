from __future__ import annotations

from pathlib import Path


def build_imagefolder(root: str | Path, split: str, transform=None):
    try:
        from torchvision.datasets import ImageFolder
    except ImportError as e:
        raise RuntimeError("torchvision is required for training") from e
    path = Path(root) / split
    if not path.exists():
        raise FileNotFoundError(path)
    return ImageFolder(str(path), transform=transform)
