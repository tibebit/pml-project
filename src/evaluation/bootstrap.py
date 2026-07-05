from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.uncertainty import compute_predictive_entropy


DEFAULT_BOOTSTRAP_METRICS = (
    "log_loss",
    "brier_score",
    "balanced_accuracy",
    "mean_predictive_entropy",
)


def bootstrap_metric_intervals(
    predictions_df: pd.DataFrame,
    n_repeats: int = 1000,
    seed: int = RANDOM_SEED,
    confidence_level: float = 0.95,
    metric_names: tuple[str, ...] = DEFAULT_BOOTSTRAP_METRICS,
) -> pd.DataFrame:
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    _validate_singer_predictions(predictions_df)

    estimates = _compute_metrics(predictions_df)
    unknown = sorted(set(metric_names) - set(estimates))
    if unknown:
        raise ValueError(f"Unknown metric names: {unknown}")

    rng = np.random.default_rng(seed)
    singer_ids = predictions_df["singer_id"].to_numpy()
    indexed = predictions_df.set_index("singer_id", drop=False)
    bootstrap_values = {metric_name: [] for metric_name in metric_names}

    for _ in range(n_repeats):
        # Resample singers with replacement
        sampled_ids = rng.choice(singer_ids, size=len(singer_ids), replace=True)
        sample_df = indexed.loc[sampled_ids].reset_index(drop=True)
        sample_metrics = _compute_metrics(sample_df)
        for metric_name in metric_names:
            bootstrap_values[metric_name].append(sample_metrics[metric_name])

    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    rows = []
    for metric_name in metric_names:
        values = np.asarray(bootstrap_values[metric_name], dtype=float)
        estimate = float(estimates[metric_name])
        lower = float(np.quantile(values, lower_q))
        upper = float(np.quantile(values, upper_q))
        rows.append(
            {
                "metric": metric_name,
                "estimate": estimate,
                "lower": min(lower, estimate),
                "upper": max(upper, estimate),
                "confidence_level": float(confidence_level),
                "bootstrap_repeats": int(n_repeats),
                "n_singers": int(len(singer_ids)),
            }
        )

    return pd.DataFrame.from_records(rows)


def _compute_metrics(predictions_df: pd.DataFrame) -> dict[str, float]:
    metrics = compute_binary_metrics(predictions_df["y_true"], predictions_df["p_dramatic"])
    entropies = [
        compute_predictive_entropy(1.0 - float(probability), float(probability))
        for probability in predictions_df["p_dramatic"]
    ]
    metrics["mean_predictive_entropy"] = float(np.mean(entropies))
    return metrics


def _validate_singer_predictions(predictions_df: pd.DataFrame) -> None:
    _require_columns(predictions_df, ["singer_id", "y_true", "p_dramatic"])
    if predictions_df.empty:
        raise ValueError("Cannot bootstrap empty predictions")
    duplicated = predictions_df["singer_id"].duplicated()
    if duplicated.any():
        examples = predictions_df.loc[duplicated, "singer_id"].tolist()[:5]
        raise ValueError(f"Predictions must be singer-level; duplicated singer_id values: {examples}")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
