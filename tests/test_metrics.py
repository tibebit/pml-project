import math

import pytest

from src.evaluation.metrics import compute_binary_metrics


def test_compute_binary_metrics_are_finite_and_in_range():
    metrics = compute_binary_metrics([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.7])

    assert math.isfinite(metrics["log_loss"])
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_compute_binary_metrics_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.2, 1.2])
