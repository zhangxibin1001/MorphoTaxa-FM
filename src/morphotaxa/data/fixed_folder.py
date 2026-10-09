from __future__ import annotations

from pathlib import Path
from typing import Callable
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def discover_classes(train_dir: str | Path) -> tuple[list[str], dict[str, int]]:
    train_dir = Path(train_dir)
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found under {train_dir}")
    return classes, {name: i for i, name in enumerate(classes)}


class FixedClassFolder:
    """Folder dataset using a caller-provided class mapping for every split.

    This prevents a val/test split with a missing class folder from silently
    changing class IDs relative to train.
    """

    def __init__(self, split_dir: str | Path, class_to_idx: dict[str, int], transform: Callable | None = None):
        from torch.utils.data import Dataset
        self._dataset_base = Dataset
        self.split_dir = Path(split_dir)
        self.class_to_idx = dict(class_to_idx)
        self.classes = [x for x, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]
        self.transform = transform
        samples: list[tuple[str, int]] = []
        unknown = []
        if not self.split_dir.exists():
            raise FileNotFoundError(self.split_dir)
        for class_dir in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            if class_dir.name not in self.class_to_idx:
                unknown.append(class_dir.name)
                continue
            target = self.class_to_idx[class_dir.name]
            for p in sorted(class_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    samples.append((str(p), target))
        if unknown:
            raise RuntimeError(f"Split {self.split_dir} contains unknown classes: {unknown[:10]}")
        if not samples:
            raise RuntimeError(f"No images found in {self.split_dir}")
        self.samples = samples
        self.targets = [y for _, y in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path
