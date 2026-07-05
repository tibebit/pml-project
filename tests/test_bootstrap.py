import pandas as pd

from src.evaluation.bootstrap import bootstrap_metric_intervals


def test_bootstrap_intervals_contain_estimates():
    predictions = pd.DataFrame(
        {
            "singer_id": [f"s{i}" for i in range(8)],
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
            "p_dramatic": [0.10, 0.80, 0.25, 0.70, 0.40, 0.65, 0.20, 0.90],
        }
    )

    intervals = bootstrap_metric_intervals(predictions, n_repeats=30, seed=11)

    assert {"log_loss", "brier_score", "balanced_accuracy"} <= set(intervals["metric"])
    assert (intervals["lower"] <= intervals["estimate"]).all()
    assert (intervals["estimate"] <= intervals["upper"]).all()
    assert set(intervals["bootstrap_repeats"]) == {30}
