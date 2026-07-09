from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.runtime_env import configure_model_runtime_env

configure_model_runtime_env()

import arviz as az


def plot_prior_predictive_check(
    idata,
    output_path: str | Path,
    feature_names: list[str] | tuple[str, ...],
) -> Path:
    prior_x = np.asarray(idata.prior_predictive["x"].sel(feature=feature_names).values, dtype=float)
    observed_x = np.asarray(idata.observed_data["x"].sel(feature=feature_names).values, dtype=float)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(feature_names), figsize=(5.0 * len(feature_names), 4.0))
    axes = np.atleast_1d(axes)
    for axis, feature_name, feature_prior, feature_observed in zip(
        axes,
        feature_names,
        np.moveaxis(prior_x, -1, 0),
        np.moveaxis(observed_x, -1, 0),
    ):
        prior_values = feature_prior.reshape(-1)
        observed_values = feature_observed.reshape(-1)
        axis.hist(prior_values, bins=50, density=True, alpha=0.45, label="prior predictive")
        axis.hist(observed_values, bins=50, density=True, alpha=0.45, label="observed")
        axis.set_title(feature_name)
        axis.set_xlabel("standardized value")
        axis.set_ylabel("density")
        axis.legend()

    fig.suptitle("Prior predictive check")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def summarize_mcmc_diagnostics_summary(
    idata,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
) -> pd.DataFrame:
    summary = _arviz_parameter_summary(idata)
    divergences = _count_divergences(idata)
    bfmi = _bfmi_values(idata)
    max_rhat = float(summary["r_hat"].max(skipna=True))
    min_bulk_ess = float(summary["ess_bulk"].min(skipna=True))
    min_tail_ess = float(summary["ess_tail"].min(skipna=True))
    min_bfmi = float(np.nanmin(bfmi)) if bfmi.size else np.nan
    # warning flag for final reporting
    sampler_warnings = bool(
        divergences > 0
        or (bfmi.size and min_bfmi < 0.3)
        or max_rhat > 1.01
        or min_tail_ess < 100
    )
    return pd.DataFrame.from_records(
        [
            {
                "chains": int(chains),
                "draws": int(draws),
                "tune": int(tune),
                "target_accept": float(target_accept),
                "n_parameters_reported": int(len(summary)),
                "n_divergences": int(divergences),
                "max_rhat": max_rhat,
                "min_bulk_ess": min_bulk_ess,
                "min_tail_ess": min_tail_ess,
                "min_bfmi": min_bfmi,
                "sampler_warnings": sampler_warnings,
            }
        ]
    )


def _arviz_parameter_summary(idata) -> pd.DataFrame:
    return az.summary(
        idata,
        var_names=["pi", "mu", "tau", "sigma", "z"],
        filter_vars="like",
    )


def _count_divergences(idata) -> int:
    if not hasattr(idata, "sample_stats") or "diverging" not in idata.sample_stats:
        return 0
    return int(idata.sample_stats["diverging"].sum().item())


def _bfmi_values(idata) -> np.ndarray:
    try:
        return np.asarray(az.bfmi(idata), dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
