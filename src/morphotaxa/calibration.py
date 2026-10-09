from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(logits: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = _softmax(logits)
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(_softmax(logits), 1e-12, 1.0)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def multiclass_brier(logits: np.ndarray, y: np.ndarray) -> float:
    p = _softmax(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
