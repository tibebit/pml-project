import pandas as pd
import pytest

from src.config import COVARIANCE, MODEL_FEATURE_COLUMNS, MODEL_VERSION, MODEL_VOICE_TYPES
from src.dataset.checks import ExtractedVocalizationError
from src.models.hierarchical_pymc import build_hierarchical_model, prepare_hierarchical_data
from src.models.priors import PriorConfig


def test_model_contract_constants():
    assert MODEL_VERSION == "hierarchical_diagonal_v1"
    assert COVARIANCE == "diagonal"
    assert MODEL_FEATURE_COLUMNS == ["PHE", "SC"]
    assert MODEL_VOICE_TYPES == ("soprano", "tenor", "baritone", "bass")


def test_prepare_hierarchical_data_shapes_and_voice_type_scope():
    df = _tiny_df()

    model_data = prepare_hierarchical_data(df, ["PHE", "SC"])

    assert model_data.voice_types == MODEL_VOICE_TYPES
    assert "bass_dramatic_m1" in model_data.singer_ids
    assert model_data.class_labels == ("lyric", "dramatic")
    assert model_data.x.shape == (7, 2)
    assert model_data.n_singers == 7
    assert model_data.singer_voice_idx.shape == (7,)
    assert model_data.singer_class_idx.shape == (7,)
    assert model_data.vocalization_singer_idx.shape == (7,)
    assert model_data.vocalization_voice_idx.shape == (7,)


def test_prepare_hierarchical_data_rejects_identifier_features():
    df = _tiny_df()

    with pytest.raises(ExtractedVocalizationError):
        prepare_hierarchical_data(df, ["PHE", "singer_id"])


def test_build_hierarchical_model_contains_v1_variables_and_dims():
    model_data = prepare_hierarchical_data(_tiny_df(), ["PHE", "SC"])

    model = build_hierarchical_model(model_data, PriorConfig())

    assert {"pi", "mu", "tau", "sigma", "z", "x", "c_obs"} <= set(model.named_vars)
    assert model.coords["voice_type"] == MODEL_VOICE_TYPES
    assert model.coords["class_label"] == ("lyric", "dramatic")
    assert model.coords["feature"] == ("PHE", "SC")


def _tiny_df():
    records = []
    for voice_type, prefix in [("soprano", "s"), ("tenor", "t"), ("baritone", "b")]:
        for class_label, class_id in [("lyric", 0), ("dramatic", 1)]:
            records.append(
                {
                    "sample_id": f"{prefix}_{class_label}_1",
                    "singer_id": f"{voice_type}_{class_label}_1",
                    "voice_type": voice_type,
                    "class_label": class_label,
                    "class_id": class_id,
                    "PHE": 10.0 + class_id * 5.0,
                    "SC": 1000.0 + class_id * 100.0,
                }
            )
    records.append(
        {
            "sample_id": "bass_1",
            "singer_id": "bass_dramatic_m1",
            "voice_type": "bass",
            "class_label": "dramatic",
            "class_id": 1,
            "PHE": 22.0,
            "SC": 2400.0,
        }
    )
    return pd.DataFrame.from_records(records)
