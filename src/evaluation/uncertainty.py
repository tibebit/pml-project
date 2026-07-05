from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal
from typing import Union

import numpy as np
import pandas as pd

from src.models.posterior_predictive import predict_new_singer


MValue = Union[int, Literal["all"]]
DEFAULT_M_VALUES: tuple[MValue, ...] = (1, 2, 4, 8, "all")


def compute_predictive_entropy(p_lyric: float, p_dramatic: float) -> float:
    probabilities = np.asarray([p_lyric, p_dramatic], dtype=float)
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Class probabilities must be in [0, 1]")
    total = probabilities.sum()
    if not np.isclose(total, 1.0):
        raise ValueError("Class probabilities must sum to 1")
    positive = probabilities[probabilities > 0.0]
    # Natural-log entropy: max binary uncertainty is log(2), about 0.693.
    return float(-np.sum(positive * np.log(positive)))


def true_class_probability(class_label: str, p_lyric: float, p_dramatic: float) -> float:
    if class_label == "lyric":
        return float(p_lyric)
    if class_label == "dramatic":
        return float(p_dramatic)
    raise ValueError(f"Unknown class_label: {class_label}")


def predict_for_vocalization_counts(
    idata,
    df: pd.DataFrame,
    metadata: dict[str, object],
    m_values: Sequence[MValue] = DEFAULT_M_VALUES,
    predictor: Callable = predict_new_singer,
) -> pd.DataFrame:
    _require_columns(df, ["singer_id", "voice_type", "class_label"])
    feature_columns = list(metadata["feature_columns"])
    _require_columns(df, feature_columns)
    _validate_m_values(m_values)

    rows = []
    for singer_id, singer_df in df.groupby("singer_id", sort=True):
        voice_type = _single_value(singer_df, "voice_type", singer_id)
        class_label = _single_value(singer_df, "class_label", singer_id)
        sorted_singer_df = _sort_vocalizations(singer_df)

        for m_requested in m_values:
            # Deterministic subsets make the uncertainty experiment reproducible.
            selected = _select_vocalizations(sorted_singer_df, m_requested)
            observations = selected[feature_columns].copy()
            prediction = predictor(
                idata=idata,
                observations=observations,
                voice_type=voice_type,
                metadata=metadata,
            )
            p_lyric = float(prediction["p_lyric"])
            p_dramatic = float(prediction["p_dramatic"])
            rows.append(
                {
                    "singer_id": singer_id,
                    "voice_type": voice_type,
                    "class_label": class_label,
                    "m_requested": str(m_requested),
                    "n_vocalizations_used": int(len(selected)),
                    "p_lyric": p_lyric,
                    "p_dramatic": p_dramatic,
                    "p_true_class": true_class_probability(class_label, p_lyric, p_dramatic),
                    "predictive_entropy": compute_predictive_entropy(p_lyric, p_dramatic),
                }
            )

    return pd.DataFrame.from_records(rows)


def summarize_uncertainty_by_m(results_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(results_df, ["m_requested", "p_true_class", "predictive_entropy"])
    return (
        results_df.groupby("m_requested", sort=False)
        .agg(
            n_singers=("singer_id", "nunique"),
            mean_p_true_class=("p_true_class", "mean"),
            median_p_true_class=("p_true_class", "median"),
            mean_predictive_entropy=("predictive_entropy", "mean"),
            median_predictive_entropy=("predictive_entropy", "median"),
            q25_predictive_entropy=("predictive_entropy", lambda values: values.quantile(0.25)),
            q75_predictive_entropy=("predictive_entropy", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )


def plot_uncertainty_vs_m(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize_uncertainty_by_m(results_df)
    fig, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.plot(summary["m_requested"], summary["mean_predictive_entropy"], marker="o")
    axis.set_xlabel("vocalizations used")
    axis.set_ylabel("mean predictive entropy")
    axis.set_title("Uncertainty vs vocalizations")
    axis.grid(alpha=0.25)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _select_vocalizations(df: pd.DataFrame, m_requested: MValue) -> pd.DataFrame:
    if m_requested == "all":
        return df.copy()
    return df.head(int(m_requested)).copy()


def _sort_vocalizations(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" in df.columns:
        return df.sort_values("sample_id", kind="mergesort")
    return df.sort_index(kind="mergesort")


def _single_value(df: pd.DataFrame, column: str, singer_id: str) -> str:
    values = df[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Inconsistent {column} for singer_id {singer_id}")
    return str(values.iloc[0])


def _validate_m_values(m_values: Sequence[MValue]) -> None:
    if not m_values:
        raise ValueError("At least one m value is required")
    for value in m_values:
        if value == "all":
            continue
        if not isinstance(value, int) or value < 1:
            raise ValueError("m values must be positive integers or 'all'")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
