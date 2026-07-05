import numpy as np
import pandas as pd
import pytest

from src.evaluation.final_evaluation import (
    compute_final_metrics,
    compute_test_uncertainty_by_m,
    make_evaluation_split,
    predict_test_singers,
)
from src.config import MODEL_VOICE_TYPES
from src.models.hierarchical_pymc import prepare_hierarchical_data
from src.models.posterior_predictive import build_model_metadata
from src.splits.singer_split import check_no_singer_leakage


def test_evaluation_split_is_singer_disjoint():
    train_df, test_df = make_evaluation_split(_example_vocalization_df(), test_size=0.5, seed=3)

    check_no_singer_leakage(train_df, test_df)
    assert set(train_df["singer_id"]).isdisjoint(set(test_df["singer_id"]))


def test_evaluation_split_rejects_unknown_voice_types():
    df = _example_vocalization_df()
    df.loc[0, "voice_type"] = "unknown"

    with pytest.raises(ValueError, match="Unexpected voice_type"):
        make_evaluation_split(df, test_size=0.5, seed=3)


def test_scaler_metadata_comes_from_train_data_only():
    df = _example_vocalization_df()
    train_df, _ = make_evaluation_split(df, test_size=0.5, seed=3)

    model_data = prepare_hierarchical_data(
        train_df,
        ["PHE", "SC"],
        voice_types=MODEL_VOICE_TYPES,
    )
    metadata = build_model_metadata(model_data)

    train_x = train_df.sort_values(["singer_id", "sample_id"])[["PHE", "SC"]].to_numpy(dtype=float)
    np.testing.assert_allclose(metadata["feature_means"], train_x.mean(axis=0))
    np.testing.assert_allclose(metadata["feature_scales"], train_x.std(axis=0))


def test_predict_test_singers_does_not_pass_singer_id_as_feature():
    _, test_df = make_evaluation_split(_example_vocalization_df(), test_size=0.5, seed=3)

    predictions = predict_test_singers(
        idata=None,
        test_df=test_df,
        metadata=_metadata(),
        predictor=_assert_feature_only_predictor,
    )

    assert predictions["singer_id"].is_unique
    assert predictions["n_vocalizations"].min() == 2
    assert predictions["p_dramatic"].between(0.0, 1.0).all()


def test_final_metrics_reject_sample_level_duplicate_singers():
    predictions = pd.DataFrame(
        {
            "singer_id": ["s1", "s1", "s2"],
            "y_true": [0, 0, 1],
            "p_dramatic": [0.2, 0.3, 0.8],
        }
    )

    with pytest.raises(ValueError, match="singer-level"):
        compute_final_metrics(predictions)


def test_final_metrics_are_compact_for_small_v1_holdout():
    predictions = pd.DataFrame(
        {
            "singer_id": ["s1", "s2", "s3", "s4"],
            "y_true": [0, 1, 0, 1],
            "p_dramatic": [0.2, 0.8, 0.4, 0.7],
        }
    )

    metrics = compute_final_metrics(predictions)

    assert list(metrics.columns) == [
        "model",
        "log_loss",
        "brier_score",
        "balanced_accuracy",
        "n_singers",
    ]


def test_uncertainty_by_m_uses_test_singers_only():
    train_df, test_df = make_evaluation_split(_example_vocalization_df(), test_size=0.5, seed=3)

    result = compute_test_uncertainty_by_m(
        idata=None,
        test_df=test_df,
        metadata=_metadata(),
        m_values=(1, "all"),
        predictor=_fake_predictor,
    )

    assert set(result["singer_id"]) == set(test_df["singer_id"].unique())
    assert set(result["singer_id"]).isdisjoint(set(train_df["singer_id"].unique()))


def _fake_predictor(idata, observations, voice_type, metadata):
    del idata, observations, voice_type, metadata
    return {"p_lyric": 0.35, "p_dramatic": 0.65}


def _assert_feature_only_predictor(idata, observations, voice_type, metadata):
    del idata, voice_type, metadata
    assert list(observations.columns) == ["PHE", "SC"]
    assert "singer_id" not in observations.columns
    return {"p_lyric": 0.35, "p_dramatic": 0.65}


def _metadata():
    return {"feature_columns": ["PHE", "SC"]}


def _example_vocalization_df():
    records = []
    voice_types = ["soprano", "tenor", "baritone", "bass"]
    class_pairs = [("lyric", 0), ("dramatic", 1)]
    for voice_index, voice_type in enumerate(voice_types):
        for class_label, class_id in class_pairs:
            for singer_index in range(2):
                singer_id = f"{voice_type}_{class_label}_{singer_index}"
                for sample_index in range(2):
                    records.append(
                        {
                            "sample_id": f"{singer_id}_{sample_index}",
                            "singer_id": singer_id,
                            "voice_type": voice_type,
                            "class_label": class_label,
                            "class_id": class_id,
                            "PHE": 10.0 * voice_index + 2.0 * class_id + singer_index + sample_index,
                            "SC": 100.0 * voice_index + 20.0 * class_id + 5.0 * singer_index + sample_index,
                        }
                    )
    return pd.DataFrame.from_records(records)
