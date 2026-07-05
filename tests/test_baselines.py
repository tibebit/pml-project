import pandas as pd
import pytest

from src.models.logistic import fit_logistic, predict_logistic
from src.models.prevalence import fit_prevalence, predict_prevalence
from src.models.random_forest import fit_random_forest, predict_random_forest


FEATURE_COLS = ["mean_PHE", "mean_SC", "sd_PHE", "sd_SC"]


def test_prevalence_predictions_are_constant():
    train_df = _example_model_df()
    test_df = train_df.iloc[:3].copy()

    model = fit_prevalence(train_df)
    predictions = predict_prevalence(model, test_df)

    assert predictions.nunique() == 1
    assert predictions.iloc[0] == pytest.approx(train_df["class_id"].mean())


def test_logistic_predictions_are_probabilities():
    train_df = _example_model_df()
    test_df = train_df.iloc[:4].copy()

    model = fit_logistic(train_df, FEATURE_COLS, seed=1)
    predictions = predict_logistic(model, test_df, FEATURE_COLS)

    assert predictions.between(0.0, 1.0).all()


def test_random_forest_predictions_are_probabilities():
    train_df = _example_model_df()
    test_df = train_df.iloc[:4].copy()

    model = fit_random_forest(train_df, FEATURE_COLS, seed=1)
    predictions = predict_random_forest(model, test_df, FEATURE_COLS)

    assert predictions.between(0.0, 1.0).all()


def test_model_features_reject_identifier_columns():
    train_df = _example_model_df()

    with pytest.raises(ValueError):
        fit_logistic(train_df, ["mean_PHE", "singer_id"], seed=1)


def _example_model_df():
    return pd.DataFrame(
        {
            "singer_id": [f"s{i}" for i in range(8)],
            "class_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "mean_PHE": [10, 30, 12, 34, 14, 36, 16, 38],
            "mean_SC": [1000, 1300, 1020, 1340, 1040, 1360, 1060, 1380],
            "sd_PHE": [1.0, 1.4, 1.1, 1.5, 1.2, 1.6, 1.3, 1.7],
            "sd_SC": [20, 30, 22, 32, 24, 34, 26, 36],
        }
    )
