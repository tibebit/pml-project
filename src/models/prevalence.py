from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PrevalenceModel:
    p_dramatic: float


def fit_prevalence(train_df: pd.DataFrame) -> PrevalenceModel:
    _require_columns(train_df, ["class_id"])
    if train_df.empty:
        raise ValueError("Cannot fit prevalence baseline on an empty training set")
    # class_id is lyric=0 and dramatic=1, so the mean is the dramatic fraction.
    return PrevalenceModel(p_dramatic=float(train_df["class_id"].mean()))


def predict_prevalence(model: PrevalenceModel, test_df: pd.DataFrame) -> pd.Series:
    return pd.Series(model.p_dramatic, index=test_df.index, name="p_dramatic")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
