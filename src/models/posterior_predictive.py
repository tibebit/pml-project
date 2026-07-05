from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import COVARIANCE, MODEL_VERSION

if TYPE_CHECKING:
    from src.models.hierarchical_pymc import HierarchicalModelData


_EPS = 1e-12


def build_model_metadata(model_data: HierarchicalModelData) -> dict[str, object]:
    # Posterior draws need this non-random context to score unseen singers.
    return {
        "model_version": MODEL_VERSION,
        "covariance": COVARIANCE,
        "feature_columns": list(model_data.feature_cols),
        "feature_means": model_data.feature_means.astype(float).tolist(),
        "feature_scales": model_data.feature_scales.astype(float).tolist(),
        "voice_types": list(model_data.voice_types),
        "class_labels": list(model_data.class_labels),
    }


def standardize_new_observations(
    observations: pd.DataFrame,
    feature_columns: list[str],
    feature_means: list[float] | np.ndarray,
    feature_scales: list[float] | np.ndarray,
) -> np.ndarray:
    missing = [column for column in feature_columns if column not in observations.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if observations.empty:
        raise ValueError("At least one vocalization is required")

    x = observations[feature_columns].astype(float).to_numpy()
    means = np.asarray(feature_means, dtype=float)
    scales = np.asarray(feature_scales, dtype=float)
    if means.shape[0] != len(feature_columns) or scales.shape[0] != len(feature_columns):
        raise ValueError("feature_means and feature_scales must match feature_columns")
    if np.any(scales == 0.0):
        raise ValueError("feature_scales must be non-zero")
    
    return (x - means) / scales


def compute_class_log_scores(
    idata,
    standardized_observations: np.ndarray,
    voice_type: str,
    voice_types: list[str],
    class_labels: list[str],
    feature_columns: list[str],
) -> dict[str, float]:
    if voice_type not in voice_types:
        raise ValueError(f"Unknown voice_type {voice_type!r}. Known voice types: {voice_types}")
    if standardized_observations.ndim != 2:
        raise ValueError("standardized_observations must be a 2D array")
    if standardized_observations.shape[1] != len(feature_columns):
        raise ValueError("standardized_observations feature dimension does not match feature_columns")

    posterior = idata.posterior
    pi_samples = _stack_samples(posterior["pi"].sel(voice_type=voice_type))
    sigma_samples = _stack_samples(
        posterior["sigma"].sel(voice_type=voice_type, feature=feature_columns)
    )

    log_scores: dict[str, float] = {}
    for class_label in class_labels:
        # Candidate-class parameters from each posterior draw.
        mu_samples = _stack_samples(
            posterior["mu"].sel(
                voice_type=voice_type,
                class_label=class_label,
                feature=feature_columns,
            )
        )
        tau_samples = _stack_samples(
            posterior["tau"].sel(
                voice_type=voice_type,
                class_label=class_label,
                feature=feature_columns,
            )
        )
        log_prior = _class_log_prior(pi_samples, class_label)
        log_likelihood = _normal_normal_log_likelihood_by_draw(
            standardized_observations,
            mu_samples,
            tau_samples,
            sigma_samples,
        )
        # Monte Carlo version of posterior averaging
        log_scores[class_label] = float(_logsumexp(log_prior + log_likelihood) - np.log(len(log_prior)))
    return log_scores


def normalize_class_scores(class_log_scores: dict[str, float]) -> dict[str, float]:
    # Normalization of class log scores.
    denominator = _logsumexp(np.asarray(list(class_log_scores.values()), dtype=float))
    return {
        class_label: float(np.exp(log_score - denominator))
        for class_label, log_score in class_log_scores.items()
    }


def predict_new_singer(
    idata,
    observations: pd.DataFrame,
    voice_type: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    feature_columns = list(metadata["feature_columns"])
    standardized = standardize_new_observations(
        observations,
        feature_columns=feature_columns,
        feature_means=metadata["feature_means"],
        feature_scales=metadata["feature_scales"],
    )
    class_log_scores = compute_class_log_scores(
        idata=idata,
        standardized_observations=standardized,
        voice_type=voice_type,
        voice_types=list(metadata["voice_types"]),
        class_labels=list(metadata["class_labels"]),
        feature_columns=feature_columns,
    )
    class_probabilities = normalize_class_scores(class_log_scores)
    return {
        "voice_type": voice_type,
        "n_vocalizations": int(len(observations)),
        "class_log_scores": class_log_scores,
        "class_probabilities": class_probabilities,
        "p_dramatic": class_probabilities["dramatic"],
        "p_lyric": class_probabilities["lyric"],
    }


def _stack_samples(data_array) -> np.ndarray:
    stacked = data_array.stack(sample=("chain", "draw")).transpose("sample", ...)
    return np.asarray(stacked.values, dtype=float)


def _class_log_prior(pi_samples: np.ndarray, class_label: str) -> np.ndarray:
    pi = np.clip(pi_samples, _EPS, 1.0 - _EPS)
    if class_label == "dramatic":
        return np.log(pi)
    if class_label == "lyric":
        return np.log1p(-pi)
    raise ValueError(f"Unsupported class_label: {class_label}")


def _normal_normal_log_likelihood_by_draw(
    x: np.ndarray,
    mu_samples: np.ndarray,
    tau_samples: np.ndarray,
    sigma_samples: np.ndarray,
) -> np.ndarray:
    n_observations, n_features = x.shape
    n_draws = mu_samples.shape[0]
    log_likelihood = np.zeros(n_draws, dtype=float)

    tau = np.maximum(tau_samples, _EPS)
    sigma = np.maximum(sigma_samples, _EPS)
    for feature_idx in range(n_features):
        residuals = x[:, feature_idx][None, :] - mu_samples[:, feature_idx][:, None]
        sigma2 = sigma[:, feature_idx] ** 2
        tau2 = tau[:, feature_idx] ** 2
        # After integrating out z_*, vocalizations from the same singer become
        # correlated: sigma^2 contributes independent noise, tau^2 contributes a
        # shared covariance term across all pairs of vocalizations.
        sum_sq = np.sum(residuals**2, axis=1)
        sum_residual = np.sum(residuals, axis=1)
        log_det = n_observations * np.log(sigma2) + np.log1p(n_observations * tau2 / sigma2)
        quad = (sum_sq / sigma2) - (
            tau2 * sum_residual**2 / (sigma2 * (sigma2 + n_observations * tau2))
        )
        log_likelihood += -0.5 * (n_observations * np.log(2.0 * np.pi) + log_det + quad)
    return log_likelihood


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))
