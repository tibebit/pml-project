from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.config import MODEL_VOICE_TYPES
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.uncertainty import (
    DEFAULT_M_VALUES,
    MValue,
    predict_for_vocalization_counts,
    true_class_probability,
)
from src.models.posterior_predictive import predict_new_singer
from src.splits.singer_split import check_no_singer_leakage, make_final_holdout_split


def make_evaluation_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(df, ["singer_id", "voice_type", "class_label"])
    unknown_voice_types = sorted(set(df["voice_type"]) - set(MODEL_VOICE_TYPES))
    if unknown_voice_types:
        raise ValueError(f"Unexpected voice_type values: {unknown_voice_types}")
    eval_df = df.copy()
    if eval_df.empty:
        raise ValueError("No rows available for evaluation")
    train_df, test_df = make_final_holdout_split(eval_df, test_size=test_size, seed=seed)
    check_no_singer_leakage(train_df, test_df)
    return train_df, test_df


def predict_test_singers(
    idata,
    test_df: pd.DataFrame,
    metadata: dict[str, object],
    predictor: Callable = predict_new_singer,
) -> pd.DataFrame:
    feature_columns = list(metadata["feature_columns"])
    _require_columns(test_df, ["singer_id", "voice_type", "class_label", "class_id"] + feature_columns)

    rows = []
    for singer_id, singer_df in test_df.groupby("singer_id", sort=True):
        voice_type = _single_value(singer_df, "voice_type", singer_id)
        class_label = _single_value(singer_df, "class_label", singer_id)
        class_id = int(_single_value(singer_df, "class_id", singer_id))
        # feature columns are used for prediction
        observations = _sort_vocalizations(singer_df)[feature_columns].copy()
        prediction = predictor(
            idata=idata,
            observations=observations,
            voice_type=voice_type,
            metadata=metadata,
        )
        p_lyric = float(prediction["p_lyric"])
        p_dramatic = float(prediction["p_dramatic"])
        
        predicted_class_id = int(p_dramatic >= 0.5)
        rows.append(
            {
                "singer_id": singer_id,
                "voice_type": voice_type,
                "class_label": class_label,
                "class_id": class_id,
                "n_vocalizations": int(len(observations)),
                "p_lyric": p_lyric,
                "p_dramatic": p_dramatic,
                "p_true_class": true_class_probability(class_label, p_lyric, p_dramatic),
                "y_true": class_id,
                "predicted_class_id": predicted_class_id,
                "predicted_class_label": "dramatic" if predicted_class_id else "lyric",
            }
        )

    predictions = pd.DataFrame.from_records(rows)
    _validate_singer_level_predictions(predictions)
    return predictions


def compute_final_metrics(
    predictions_df: pd.DataFrame,
    model_name: str = "bayesian_hierarchical",
) -> pd.DataFrame:
    _validate_singer_level_predictions(predictions_df)
    metrics = compute_binary_metrics(predictions_df["y_true"], predictions_df["p_dramatic"])
    metrics["model"] = model_name
    metrics["n_singers"] = int(predictions_df["singer_id"].nunique())
    return pd.DataFrame.from_records([metrics])[
        ["model", "log_loss", "brier_score", "balanced_accuracy", "n_singers"]
    ]


def compute_test_uncertainty_by_m(
    idata,
    test_df: pd.DataFrame,
    metadata: dict[str, object],
    m_values: Sequence[MValue] = DEFAULT_M_VALUES,
    predictor: Callable = predict_new_singer,
) -> pd.DataFrame:
    # Re-predict the same held-out singers as more vocalizations become available.
    return predict_for_vocalization_counts(
        idata=idata,
        df=test_df,
        metadata=metadata,
        m_values=m_values,
        predictor=predictor,
    )


def summarize_final_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, split_df in (("train", train_df), ("test", test_df)):
        grouped = (
            split_df.groupby(["voice_type", "class_label"], sort=True)
            .agg(n_singers=("singer_id", "nunique"), n_vocalizations=("singer_id", "size"))
            .reset_index()
        )
        grouped.insert(0, "split", split_name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _validate_singer_level_predictions(predictions_df: pd.DataFrame) -> None:
    _require_columns(predictions_df, ["singer_id", "y_true", "p_dramatic"])
    if predictions_df.empty:
        raise ValueError("Predictions are empty")
    duplicated = predictions_df["singer_id"].duplicated()
    if duplicated.any():
        examples = predictions_df.loc[duplicated, "singer_id"].tolist()[:5]
        raise ValueError(f"Predictions must be singer-level; duplicated singer_id values: {examples}")
    probabilities = predictions_df["p_dramatic"].astype(float)
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("p_dramatic values must be in [0, 1]")


def _sort_vocalizations(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" in df.columns:
        return df.sort_values("sample_id", kind="mergesort")
    return df.sort_index(kind="mergesort")


def _single_value(df: pd.DataFrame, column: str, singer_id: str) -> str:
    values = df[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Inconsistent {column} for singer_id {singer_id}")
    return str(values.iloc[0])


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
