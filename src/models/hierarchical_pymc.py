from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.runtime_env import configure_model_runtime_env

configure_model_runtime_env()

import pymc as pm

from src.config import MODEL_VOICE_TYPES, RANDOM_SEED
from src.dataset.checks import check_no_identifier_features
from src.models.priors import PriorConfig


CLASS_LABELS = ("lyric", "dramatic")


@dataclass(frozen=True)
class HierarchicalModelData:
    x: np.ndarray
    feature_cols: tuple[str, ...]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    voice_types: tuple[str, ...]
    class_labels: tuple[str, ...]
    singer_ids: tuple[str, ...]
    singer_voice_idx: np.ndarray
    singer_class_idx: np.ndarray
    vocalization_singer_idx: np.ndarray
    vocalization_voice_idx: np.ndarray

    @property
    def n_voice_types(self) -> int:
        return len(self.voice_types)

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_singers(self) -> int:
        return len(self.singer_ids)

    @property
    def n_vocalizations(self) -> int:
        return int(self.x.shape[0])


def prepare_hierarchical_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    voice_types: tuple[str, ...] | list[str] | None = None,
) -> HierarchicalModelData:
    check_no_identifier_features(feature_cols)
    voice_types = tuple(voice_types or MODEL_VOICE_TYPES)
    _require_columns(df, ["singer_id", "voice_type", "class_label", "class_id"] + feature_cols)

    model_df = df.loc[df["voice_type"].isin(voice_types)].copy()
    if model_df.empty:
        raise ValueError("No rows remain after voice_type filtering")

    _validate_voice_and_class_values(model_df, voice_types)
    _validate_singer_metadata_consistency(model_df)

    singer_table = (
        model_df[["singer_id", "voice_type", "class_label", "class_id"]]
        .drop_duplicates()
        .sort_values("singer_id")
        .reset_index(drop=True)
    )
    # PyMC arrays use integer indexes; coords keep singer/voice/class labels readable.
    singer_ids = tuple(singer_table["singer_id"])
    singer_lookup = {singer_id: index for index, singer_id in enumerate(singer_ids)}
    voice_lookup = {voice_type: index for index, voice_type in enumerate(voice_types)}
    class_lookup = {class_label: index for index, class_label in enumerate(CLASS_LABELS)}

    ordered_df = model_df.sort_values(["singer_id", "sample_id"]).reset_index(drop=True)
    raw_x = ordered_df[feature_cols].astype(float).to_numpy()
    # Fit standardization on the training rows only, then reuse these values for test singers.
    feature_means = raw_x.mean(axis=0)
    feature_scales = raw_x.std(axis=0)
    feature_scales = np.where(feature_scales == 0.0, 1.0, feature_scales)
    x = (raw_x - feature_means) / feature_scales

    return HierarchicalModelData(
        x=x,
        feature_cols=tuple(feature_cols),
        feature_means=feature_means,
        feature_scales=feature_scales,
        voice_types=voice_types,
        class_labels=CLASS_LABELS,
        singer_ids=singer_ids,
        singer_voice_idx=singer_table["voice_type"].map(voice_lookup).to_numpy(dtype=int),
        singer_class_idx=singer_table["class_label"].map(class_lookup).to_numpy(dtype=int),
        vocalization_singer_idx=ordered_df["singer_id"].map(singer_lookup).to_numpy(dtype=int),
        vocalization_voice_idx=ordered_df["voice_type"].map(voice_lookup).to_numpy(dtype=int),
    )


def build_hierarchical_model(
    model_data: HierarchicalModelData,
    prior_config: PriorConfig | None = None,
) -> pm.Model:
    prior = prior_config or PriorConfig()
    coords = {
        "voice_type": model_data.voice_types,
        "class_label": model_data.class_labels,
        "feature": model_data.feature_cols,
        "singer_id": model_data.singer_ids,
        "vocalization": np.arange(model_data.n_vocalizations),
    }

    with pm.Model(coords=coords) as model:
        # pi[v] = P(dramatic | voice_type=v).
        pi = pm.Beta(
            "pi",
            alpha=prior.beta_alpha,
            beta=prior.beta_beta,
            dims="voice_type",
        )
        # mu[v, c, k] is the population center for feature k.
        mu = pm.Normal(
            "mu",
            mu=0.0,
            sigma=prior.mu_scale,
            dims=("voice_type", "class_label", "feature"),
        )
        # tau[v, c, k] is between-singer variability around mu.
        tau = pm.HalfNormal(
            "tau",
            sigma=prior.tau_scale,
            dims=("voice_type", "class_label", "feature"),
        )
        # sigma[v, k] is within-singer vocalization noise.
        sigma = pm.HalfNormal(
            "sigma",
            sigma=prior.sigma_scale,
            dims=("voice_type", "feature"),
        )

        # Training labels are observed. NUTS samples pi, not c_obs.
        pm.Bernoulli(
            "c_obs",
            p=pi[model_data.singer_voice_idx],
            observed=model_data.singer_class_idx,
            dims="singer_id",
        )
        # z[s, k] is the latent acoustic signature of singer s.
        z = pm.Normal(
            "z",
            mu=mu[model_data.singer_voice_idx, model_data.singer_class_idx, :],
            sigma=tau[model_data.singer_voice_idx, model_data.singer_class_idx, :],
            dims=("singer_id", "feature"),
        )
        pm.Normal(
            "x",
            mu=z[model_data.vocalization_singer_idx, :],
            sigma=sigma[model_data.vocalization_voice_idx, :],
            observed=model_data.x,
            dims=("vocalization", "feature"),
        )
        # Independent feature-wise scales; for now we decided to not consider full covariance.

    return model


def sample_prior_predictive(
    model: pm.Model,
    draws: int = 100,
    seed: int = RANDOM_SEED,
):
    with model:
        return pm.sample_prior_predictive(samples=draws, random_seed=seed)


def sample_model(
    model: pm.Model,
    draws: int = 200,
    tune: int = 200,
    chains: int = 2,
    target_accept: float = 0.9,
    seed: int = RANDOM_SEED,
):
    with model:
        # NUTS produces posterior draws for pi, mu, tau, sigma, and z.
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_voice_and_class_values(df: pd.DataFrame, voice_types: tuple[str, ...]) -> None:
    unknown_voice_types = sorted(set(df["voice_type"]) - set(voice_types))
    if unknown_voice_types:
        raise ValueError(f"Unexpected voice_type values: {unknown_voice_types}")

    unknown_class_labels = sorted(set(df["class_label"]) - set(CLASS_LABELS))
    if unknown_class_labels:
        raise ValueError(f"Unexpected class_label values: {unknown_class_labels}")

    class_id_by_label = {"lyric": 0, "dramatic": 1}
    mismatched = df["class_id"] != df["class_label"].map(class_id_by_label)
    if mismatched.any():
        raise ValueError("class_id values do not match class_label")


def _validate_singer_metadata_consistency(df: pd.DataFrame) -> None:
    inconsistent = (
        df.groupby("singer_id")[["voice_type", "class_label", "class_id"]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        examples = inconsistent[inconsistent].index.tolist()[:5]
        raise ValueError(f"Inconsistent singer metadata for singer_id values: {examples}")
