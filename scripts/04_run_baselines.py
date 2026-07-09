from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_FEATURE_COLUMNS, MODEL_VOICE_TYPES, RANDOM_SEED
from src.dataset.tables import make_singer_level_feature_columns
from src.evaluation.bootstrap import bootstrap_metric_intervals
from src.evaluation.metrics import compute_binary_metrics
from src.models.logistic import fit_logistic, predict_logistic
from src.models.prevalence import fit_prevalence, predict_prevalence
from src.models.random_forest import fit_random_forest, predict_random_forest
from src.paths import OUTPUT_TABLES_DIR, SINGER_LEVEL_TABLE_PATH, ensure_project_dirs
from src.splits.singer_split import check_no_singer_leakage, make_final_holdout_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Run singer-level baseline models.")
    parser.add_argument("--input", type=Path, default=SINGER_LEVEL_TABLE_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--tables-dir", type=Path, default=OUTPUT_TABLES_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Singer-level table not found: {args.input}. "
            "Run python3 scripts/02_build_dataset.py first."
        )

    ensure_project_dirs()
    feature_cols = list(MODEL_FEATURE_COLUMNS)
    baseline_feature_cols = make_singer_level_feature_columns(feature_cols)
    singer_df = pd.read_csv(args.input)

    voice_types = MODEL_VOICE_TYPES
    unknown_voice_types = sorted(set(singer_df["voice_type"]) - set(voice_types))
    if unknown_voice_types:
        raise SystemExit(f"Unexpected voice_type values: {unknown_voice_types}")
    if singer_df.empty:
        raise SystemExit("No singers available for baseline evaluation")

    # The baseline split is singer-disjoint.
    train_df, test_df = make_final_holdout_split(singer_df, test_size=args.test_size, seed=args.seed)
    check_no_singer_leakage(train_df, test_df)

    predictions = _run_models(train_df, test_df, baseline_feature_cols, args.seed)
    metrics = _compute_metrics(predictions)
    bootstrap = _compute_bootstrap_intervals(predictions, args.bootstrap_repeats, args.seed)

    predictions_path = args.tables_dir / "baseline_predictions.csv"
    metrics_path = args.tables_dir / "baseline_metrics.csv"
    bootstrap_path = args.tables_dir / "baseline_bootstrap_intervals.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    print(f"Read {args.input}")
    print(f"Voice types: {list(voice_types)}")
    print(f"Train singers: {train_df['singer_id'].nunique()}")
    print(f"Test singers: {test_df['singer_id'].nunique()}")
    print(f"Features: {baseline_feature_cols}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {bootstrap_path}")
    return 0


def _run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    model_specs = [
        (
            "prevalence",
            fit_prevalence(train_df),
            lambda model: predict_prevalence(model, test_df),
        ),
        (
            "logistic",
            fit_logistic(train_df, feature_cols, seed=seed),
            lambda model: predict_logistic(model, test_df, feature_cols),
        ),
        (
            "random_forest",
            fit_random_forest(train_df, feature_cols, seed=seed),
            lambda model: predict_random_forest(model, test_df, feature_cols),
        ),
    ]

    frames = []
    base_columns = ["singer_id", "voice_type", "class_label", "class_id"]
    for model_name, model, predict in model_specs:
        model_predictions = test_df[base_columns].copy()
        model_predictions["split"] = "test"
        model_predictions["model"] = model_name
        # Shared probability column consumed by all metric functions.
        model_predictions["p_dramatic"] = predict(model).to_numpy()
        model_predictions["y_true"] = model_predictions["class_id"]
        frames.append(model_predictions)

    return pd.concat(frames, ignore_index=True)[
        ["singer_id", "voice_type", "class_label", "class_id", "split", "model", "p_dramatic", "y_true"]
    ]


def _compute_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model_name, model_df in predictions.groupby("model", sort=True):
        metrics = compute_binary_metrics(model_df["y_true"], model_df["p_dramatic"])
        metrics["model"] = model_name
        metrics["n_singers"] = int(model_df["singer_id"].nunique())
        records.append(metrics)

    return pd.DataFrame.from_records(records)[
        ["model", "log_loss", "brier_score", "balanced_accuracy", "n_singers"]
    ]


def _compute_bootstrap_intervals(
    predictions: pd.DataFrame,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    frames = []
    for model_name, model_df in predictions.groupby("model", sort=True):
        intervals = bootstrap_metric_intervals(
            model_df,
            n_repeats=n_repeats,
            seed=seed,
            metric_names=("log_loss", "brier_score", "balanced_accuracy"),
        )
        intervals.insert(0, "model", model_name)
        frames.append(intervals)

    return pd.concat(frames, ignore_index=True)[
        [
            "model",
            "metric",
            "estimate",
            "lower",
            "upper",
            "confidence_level",
            "bootstrap_repeats",
            "n_singers",
        ]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
