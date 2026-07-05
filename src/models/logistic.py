from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset.checks import check_no_identifier_features


def fit_logistic(train_df: pd.DataFrame, feature_cols: list[str], seed: int = 2026) -> Pipeline:
    _require_model_columns(train_df, feature_cols)
    model = Pipeline(
        steps=[
            # The scaler is fitted only on train data through the pipeline.
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )
    model.fit(train_df[feature_cols], train_df["class_id"])
    return model


def predict_logistic(model: Pipeline, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    _require_columns(test_df, feature_cols)
    probabilities = model.predict_proba(test_df[feature_cols])[:, 1]
    return pd.Series(probabilities, index=test_df.index, name="p_dramatic")


def _require_model_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    check_no_identifier_features(feature_cols)
    _require_columns(df, feature_cols + ["class_id"])


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
