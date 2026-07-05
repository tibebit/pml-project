from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss


def compute_binary_metrics(y_true, p_dramatic) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_dramatic, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and p_dramatic must have the same length")
    if y.shape[0] == 0:
        raise ValueError("Cannot compute metrics on empty inputs")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_dramatic values must be in [0, 1]")

    # Probability metrics use P(dramatic). Balanced accuracy uses a 0.5 threshold.
    predicted = (p >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
    }
