from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.runtime_env import configure_model_runtime_env

# Configure numerical runtime before importing PyMC-related modules.
configure_model_runtime_env()

from src.config import MODEL_FEATURE_COLUMNS, MODEL_VOICE_TYPES, RANDOM_SEED
from src.evaluation.bootstrap import bootstrap_metric_intervals
from src.evaluation.final_evaluation import (
    compute_final_metrics,
    compute_test_uncertainty_by_m,
    make_evaluation_split,
    predict_test_singers,
    summarize_final_split,
)
from src.evaluation.uncertainty import DEFAULT_M_VALUES, plot_uncertainty_vs_m, summarize_uncertainty_by_m
from src.models.diagnostics import summarize_mcmc_diagnostics_summary
from src.models.hierarchical_pymc import (
    build_hierarchical_model,
    prepare_hierarchical_data,
    sample_model,
    sample_prior_predictive,
)
from src.models.posterior_predictive import build_model_metadata
from src.models.priors import PriorConfig
from src.paths import (
    OUTPUT_FIGURES_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_TABLES_DIR,
    VOCALIZATION_TABLE_PATH,
    ensure_project_dirs,
)

MODEL_IDATA_OUTPUT = OUTPUT_MODELS_DIR / "final_train_model_idata.nc"
MODEL_METADATA_OUTPUT = OUTPUT_MODELS_DIR / "final_train_model_metadata.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final singer-disjoint Bayesian evaluation.")
    parser.add_argument("--input", type=Path, default=VOCALIZATION_TABLE_PATH)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--prior-draws", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--m-values", default="1,2,4,8,all")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=OUTPUT_TABLES_DIR / "mcmc_diagnostics_summary.csv",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=OUTPUT_TABLES_DIR / "final_metrics.csv",
    )
    parser.add_argument(
        "--uncertainty-output",
        type=Path,
        default=OUTPUT_TABLES_DIR / "uncertainty_by_m.csv",
    )
    parser.add_argument(
        "--bootstrap-output",
        type=Path,
        default=OUTPUT_TABLES_DIR / "bootstrap_intervals.csv",
    )
    parser.add_argument(
        "--uncertainty-figure-output",
        type=Path,
        default=OUTPUT_FIGURES_DIR / "test_uncertainty_vs_m.png",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Vocalization table not found: {args.input}. "
            "Run python3 scripts/02_build_dataset.py first."
        )

    ensure_project_dirs()
    df = pd.read_csv(args.input)
    feature_cols = list(MODEL_FEATURE_COLUMNS)
    voice_types = MODEL_VOICE_TYPES
    m_values = _parse_m_values(args.m_values)

    # Final evaluation is singer-disjoint
    train_df, test_df = make_evaluation_split(
        df,
        test_size=args.test_size,
        seed=args.seed,
    )
    split_summary = summarize_final_split(train_df, test_df)

    model_data = prepare_hierarchical_data(train_df, feature_cols, voice_types=voice_types)
    model = build_hierarchical_model(model_data, PriorConfig())
    prior_idata = sample_prior_predictive(model, draws=args.prior_draws, seed=args.seed)
    # NUTS samples the train posterior; test singers are scored afterwards.
    idata = sample_model(
        model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )
    idata.extend(prior_idata)

    metadata = build_model_metadata(model_data)
    run_metadata = _build_run_metadata(
        model_metadata=metadata,
        args=args,
        model_data=model_data,
        test_df=test_df,
        m_values=m_values,
    )
    idata.to_netcdf(MODEL_IDATA_OUTPUT)
    _write_json(run_metadata, MODEL_METADATA_OUTPUT)

    diagnostics = summarize_mcmc_diagnostics_summary(
        idata,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
    )
    # One prediction row per held-out singer, with posterior averaging inside the predictor
    predictions = predict_test_singers(idata, test_df, metadata)
    metrics = compute_final_metrics(predictions)
    # Bootstrap resamples singers
    bootstrap = bootstrap_metric_intervals(
        predictions,
        n_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    uncertainty = compute_test_uncertainty_by_m(
        idata=idata,
        test_df=test_df,
        metadata=metadata,
        m_values=m_values,
    )
    uncertainty_summary = summarize_uncertainty_by_m(uncertainty)

    _write_csv(metrics, args.metrics_output)
    _write_csv(bootstrap, args.bootstrap_output)
    _write_csv(uncertainty_summary, args.uncertainty_output)
    _write_csv(diagnostics, args.diagnostics_output)

    if not args.no_plots:
        plot_uncertainty_vs_m(uncertainty, args.uncertainty_figure_output)

    print(f"Read {args.input}")
    print(f"Voice types: {list(model_data.voice_types)}")
    print(f"Train singers: {model_data.n_singers}")
    print(f"Test singers: {test_df['singer_id'].nunique()}")
    print(f"Features: {list(model_data.feature_cols)}")
    print(f"m values: {[str(value) for value in m_values]}")
    print("Split summary:")
    print(split_summary.to_string(index=False))
    print(f"Wrote {MODEL_IDATA_OUTPUT}")
    print(f"Wrote {MODEL_METADATA_OUTPUT}")
    print(f"Wrote {args.diagnostics_output}")
    print(f"Wrote {args.metrics_output}")
    print(f"Wrote {args.bootstrap_output}")
    print(f"Wrote {args.uncertainty_output}")
    if not args.no_plots:
        print(f"Wrote {args.uncertainty_figure_output}")
    print(f"Divergences: {int(diagnostics['n_divergences'].iloc[0])}")
    print(f"Minimum BFMI: {diagnostics['min_bfmi'].iloc[0]}")
    if bool(diagnostics["sampler_warnings"].iloc[0]):
        print("Sampler warnings present: final reporting should inspect diagnostics before interpretation.")
    if args.draws < 500 or args.tune < 500:
        print("Smoke-run note: low draws/tune make diagnostics and metrics smoke-test quality only.")
    return 0


def _parse_m_values(text: str):
    values = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if token == "all":
            values.append("all")
        else:
            values.append(int(token))
    return tuple(values or DEFAULT_M_VALUES)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _build_run_metadata(
    model_metadata: dict[str, object],
    args: argparse.Namespace,
    model_data,
    test_df: pd.DataFrame,
    m_values,
) -> dict[str, object]:
    return {
        **model_metadata,
        "sampling": {
            "draws": int(args.draws),
            "tune": int(args.tune),
            "chains": int(args.chains),
            "target_accept": float(args.target_accept),
            "prior_draws": int(args.prior_draws),
            "seed": int(args.seed),
        },
        "evaluation": {
            "test_size": float(args.test_size),
            "train_singers": int(model_data.n_singers),
            "test_singers": int(test_df["singer_id"].nunique()),
            "m_values": [str(value) for value in m_values],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
