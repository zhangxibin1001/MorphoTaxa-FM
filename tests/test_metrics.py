import numpy as np
from morphotaxa.metrics import classification_metrics
from morphotaxa.calibration import expected_calibration_error, nll, multiclass_brier


def test_perfect_metrics():
    logits = np.array([[9., 0., 0.], [0., 9., 0.], [0., 0., 9.]])
    y = np.array([0, 1, 2])
    m = classification_metrics(logits, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert 0 <= expected_calibration_error(logits, y) <= 1
    assert nll(logits, y) >= 0
    assert multiclass_brier(logits, y) >= 0
