from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment(*paths: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = _deep_merge(cfg, load_yaml(path))
    return cfg
