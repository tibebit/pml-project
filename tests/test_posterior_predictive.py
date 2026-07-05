import arviz as az
import numpy as np
import pandas as pd
import pytest

from src.models.posterior_predictive import (
    compute_class_log_scores,
    normalize_class_scores,
    predict_new_singer,
    standardize_new_observations,
)


def test_standardization_uses_training_statistics():
    observations = pd.DataFrame({"PHE": [100.0, 110.0], "SC": [1000.0, 1200.0]})

    standardized = standardize_new_observations(
        observations,
        feature_columns=["PHE", "SC"],
        feature_means=[10.0, 100.0],
        feature_scales=[10.0, 100.0],
    )

    np.testing.assert_allclose(standardized, np.asarray([[9.0, 9.0], [10.0, 11.0]]))


def test_standardization_fails_clearly_when_feature_missing():
    observations = pd.DataFrame({"PHE": [1.0]})

    with pytest.raises(ValueError, match="Missing required feature columns"):
        standardize_new_observations(
            observations,
            feature_columns=["PHE", "SC"],
            feature_means=[0.0, 0.0],
            feature_scales=[1.0, 1.0],
        )


def test_predict_new_singer_fails_for_empty_observations():
    observations = pd.DataFrame({"PHE": [], "SC": []})

    with pytest.raises(ValueError, match="At least one vocalization is required"):
        predict_new_singer(
            idata=_fake_idata(),
            observations=observations,
            voice_type="soprano",
            metadata=_metadata(),
        )


def test_normalized_probabilities_are_valid_and_sum_to_one():
    probabilities = normalize_class_scores({"lyric": -2.0, "dramatic": -1.0})

    assert 0.0 <= probabilities["lyric"] <= 1.0
    assert 0.0 <= probabilities["dramatic"] <= 1.0
    assert probabilities["lyric"] + probabilities["dramatic"] == pytest.approx(1.0)


def test_predict_new_singer_handles_multiple_vocalizations_without_singer_id():
    idata = _fake_idata()
    metadata = _metadata()
    observations = pd.DataFrame({"PHE": [0.1, 0.2, 0.0], "SC": [0.0, 0.1, 0.2]})

    prediction = predict_new_singer(
        idata=idata,
        observations=observations,
        voice_type="soprano",
        metadata=metadata,
    )

    assert prediction["n_vocalizations"] == 3
    assert 0.0 <= prediction["p_lyric"] <= 1.0
    assert 0.0 <= prediction["p_dramatic"] <= 1.0
    assert prediction["p_lyric"] + prediction["p_dramatic"] == pytest.approx(1.0)


def test_compute_class_log_scores_fails_for_unknown_voice_type():
    with pytest.raises(ValueError, match="Unknown voice_type"):
        compute_class_log_scores(
            idata=_fake_idata(),
            standardized_observations=np.asarray([[0.0, 0.0]]),
            voice_type="bass",
            voice_types=["soprano"],
            class_labels=["lyric", "dramatic"],
            feature_columns=["PHE", "SC"],
        )


def _fake_idata():
    mu = np.zeros((1, 2, 1, 2, 2), dtype=float)
    mu[0, 0, 0, 0, :] = [-0.5, -0.5]
    mu[0, 0, 0, 1, :] = [0.5, 0.5]
    mu[0, 1, 0, 0, :] = [-0.4, -0.4]
    mu[0, 1, 0, 1, :] = [0.4, 0.4]
    coords = {
        "voice_type": ["soprano"],
        "class_label": ["lyric", "dramatic"],
        "feature": ["PHE", "SC"],
    }
    dims = {
        "pi": ["voice_type"],
        "mu": ["voice_type", "class_label", "feature"],
        "tau": ["voice_type", "class_label", "feature"],
        "sigma": ["voice_type", "feature"],
    }
    posterior = {
        "pi": np.asarray([[[0.4], [0.6]]]),
        "mu": mu,
        "tau": np.full((1, 2, 1, 2, 2), 0.7),
        "sigma": np.full((1, 2, 1, 2), 0.5),
    }
    return az.from_dict(posterior=posterior, coords=coords, dims=dims)


def _metadata():
    return {
        "feature_columns": ["PHE", "SC"],
        "feature_means": [0.0, 0.0],
        "feature_scales": [1.0, 1.0],
        "voice_types": ["soprano"],
        "class_labels": ["lyric", "dramatic"],
    }
