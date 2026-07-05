from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.dataset.checks import check_no_identifier_features


def fit_random_forest(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 2026,
) -> RandomForestClassifier:
    check_no_identifier_features(feature_cols)
    _require_columns(train_df, feature_cols + ["class_id"])
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(train_df[feature_cols], train_df["class_id"])
    return model


def predict_random_forest(
    model: RandomForestClassifier,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    _require_columns(test_df, feature_cols)
    probabilities = model.predict_proba(test_df[feature_cols])[:, 1]
    return pd.Series(probabilities, index=test_df.index, name="p_dramatic")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
