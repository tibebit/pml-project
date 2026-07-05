import pandas as pd
import pytest

from src.evaluation.uncertainty import (
    compute_predictive_entropy,
    predict_for_vocalization_counts,
    summarize_uncertainty_by_m,
    true_class_probability,
)


def test_entropy_is_near_zero_for_near_certain_probability():
    assert compute_predictive_entropy(1.0, 0.0) == pytest.approx(0.0)
    assert compute_predictive_entropy(0.0, 1.0) == pytest.approx(0.0)


def test_entropy_is_maximal_around_even_probabilities():
    assert compute_predictive_entropy(0.5, 0.5) > compute_predictive_entropy(0.9, 0.1)


def test_true_class_probability_selects_matching_probability():
    assert true_class_probability("dramatic", p_lyric=0.2, p_dramatic=0.8) == 0.8
    assert true_class_probability("lyric", p_lyric=0.7, p_dramatic=0.3) == 0.7


def test_predict_for_vocalization_counts_returns_one_row_per_singer_per_m():
    df = _example_df()

    result = predict_for_vocalization_counts(
        idata=None,
        df=df,
        metadata=_metadata(),
        m_values=(1, 2, "all"),
        predictor=_fake_predictor,
    )

    assert len(result) == df["singer_id"].nunique() * 3
    assert set(result["m_requested"]) == {"1", "2", "all"}
    assert result["n_vocalizations_used"].max() <= 3
    assert result["p_true_class"].between(0.0, 1.0).all()
    assert result["predictive_entropy"].ge(0.0).all()


def test_predict_for_vocalization_counts_uses_all_when_requested():
    result = predict_for_vocalization_counts(
        idata=None,
        df=_example_df(),
        metadata=_metadata(),
        m_values=("all",),
        predictor=_fake_predictor,
    )

    assert result.set_index("singer_id").loc["s1", "n_vocalizations_used"] == 3
    assert result.set_index("singer_id").loc["s2", "n_vocalizations_used"] == 2


def test_predictor_receives_feature_columns_only_not_singer_id():
    predict_for_vocalization_counts(
        idata=None,
        df=_example_df(),
        metadata=_metadata(),
        m_values=(1,),
        predictor=_assert_feature_only_predictor,
    )


def test_summarize_uncertainty_by_m_returns_group_summary():
    result = predict_for_vocalization_counts(
        idata=None,
        df=_example_df(),
        metadata=_metadata(),
        m_values=(1, "all"),
        predictor=_fake_predictor,
    )

    summary = summarize_uncertainty_by_m(result)

    assert list(summary["m_requested"]) == ["1", "all"]
    assert set(summary["n_singers"]) == {2}
    assert "median_p_true_class" in summary.columns
    assert "q25_predictive_entropy" in summary.columns
    assert "q75_predictive_entropy" in summary.columns


def _fake_predictor(idata, observations, voice_type, metadata):
    del idata, voice_type, metadata
    p_dramatic = min(0.95, 0.2 + 0.1 * len(observations))
    return {"p_lyric": 1.0 - p_dramatic, "p_dramatic": p_dramatic}


def _assert_feature_only_predictor(idata, observations, voice_type, metadata):
    del idata, voice_type, metadata
    assert list(observations.columns) == ["PHE", "SC"]
    assert "singer_id" not in observations.columns
    return {"p_lyric": 0.4, "p_dramatic": 0.6}


def _metadata():
    return {"feature_columns": ["PHE", "SC"]}


def _example_df():
    return pd.DataFrame(
        {
            "sample_id": ["b", "a", "c", "e", "d"],
            "singer_id": ["s1", "s1", "s1", "s2", "s2"],
            "voice_type": ["soprano", "soprano", "soprano", "tenor", "tenor"],
            "class_label": ["dramatic", "dramatic", "dramatic", "lyric", "lyric"],
            "PHE": [1.0, 2.0, 3.0, 4.0, 5.0],
            "SC": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
