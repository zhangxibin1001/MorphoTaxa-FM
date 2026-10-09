from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, optimizer=None, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "meta": meta or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
