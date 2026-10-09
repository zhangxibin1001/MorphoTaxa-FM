from __future__ import annotations

import numpy as np


def topk_accuracy(logits: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    k = min(k, logits.shape[1])
    pred = np.argpartition(logits, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(pred == y[:, None], axis=1)))


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
    pred = logits.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="macro",
        zero_division=0,
    )

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "top3": topk_accuracy(logits, y, 3),
        "top5": topk_accuracy(logits, y, 5),
    }
