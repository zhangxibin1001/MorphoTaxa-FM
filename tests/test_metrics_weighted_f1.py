import numpy as np

from sklearn.metrics import f1_score

from morphotaxa.metrics import classification_metrics


def test_weighted_f1_matches_sklearn():
    logits = np.array(
        [
            [8.0, 1.0, 0.0],
            [0.0, 8.0, 1.0],
            [0.0, 7.0, 2.0],
            [0.0, 1.0, 8.0],
            [0.0, 1.0, 8.0],
            [0.0, 8.0, 1.0],
        ],
        dtype=np.float32,
    )

    y = np.array(
        [0, 1, 2, 2, 2, 1],
        dtype=np.int64,
    )

    m = classification_metrics(
        logits,
        y,
    )

    pred = logits.argmax(axis=1)

    expected = f1_score(
        y,
        pred,
        average="weighted",
        zero_division=0,
    )

    assert "weighted_f1" in m
    assert abs(
        m["weighted_f1"] - expected
    ) < 1e-12


def test_metric_ranges():
    logits = np.array(
        [
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )

    y = np.array(
        [0, 1],
        dtype=np.int64,
    )

    m = classification_metrics(
        logits,
        y,
    )

    for key in (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top3",
        "top5",
    ):
        assert 0.0 <= m[key] <= 1.0
