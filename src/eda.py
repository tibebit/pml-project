from __future__ import annotations

import os
import tempfile
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.dataset.schema import CORE_FEATURE_COLUMNS


_TMP_CACHE_DIR = Path(tempfile.gettempdir()) / "pml_matplotlib_cache"
_TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Keep Matplotlib cache outside the repository and inside a writable directory.
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE_DIR / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CLASS_COLORS = {
    "lyric": "#2b6cb0",
    "dramatic": "#b83232",
}


def describe_features_by_voice_class(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    _require_columns(df, ["voice_type", "class_label"] + feature_cols)
    summary = (
        df.groupby(["voice_type", "class_label"], sort=True)[feature_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def compute_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [column for column in CORE_FEATURE_COLUMNS if column in df.columns]
    if len(feature_cols) < 2:
        raise ValueError("At least two core feature columns are required")

    records = []
    records.extend(_correlation_records(df, feature_cols, scope="all"))

    for voice_type, group in df.groupby("voice_type", sort=True):
        records.extend(
            _correlation_records(
                group,
                feature_cols,
                scope="voice_type",
                voice_type=voice_type,
            )
        )

    for (voice_type, class_label), group in df.groupby(["voice_type", "class_label"], sort=True):
        records.extend(
            _correlation_records(
                group,
                feature_cols,
                scope="voice_type_class",
                voice_type=voice_type,
                class_label=class_label,
            )
        )

    return pd.DataFrame.from_records(records)


def plot_feature_distributions(
    df: pd.DataFrame,
    output_path: str | Path,
    feature_cols: list[str] | None = None,
) -> Path:
    features = feature_cols or CORE_FEATURE_COLUMNS
    _require_columns(df, ["class_label"] + features)

    fig, axes = plt.subplots(1, len(features), figsize=(4.8 * len(features), 3.6), constrained_layout=True)
    if len(features) == 1:
        axes = [axes]

    for axis, feature in zip(axes, features):
        for class_label, class_df in df.groupby("class_label", sort=True):
            axis.hist(
                class_df[feature].dropna(),
                bins=24,
                alpha=0.55,
                label=class_label,
                color=CLASS_COLORS.get(class_label),
            )
        axis.set_title(feature)
        axis.set_xlabel(feature)
        axis.set_ylabel("vocalizations")
        axis.legend(frameon=False)

    return _save_figure(fig, output_path)


def plot_phe_sc_scatter(df: pd.DataFrame, output_path: str | Path) -> Path:
    _require_columns(df, ["PHE", "SC", "class_label"])
    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    _scatter_by_class(df, axis, "PHE", "SC")
    axis.set_xlabel("PHE")
    axis.set_ylabel("SC")
    axis.set_title("PHE and SC by class")
    return _save_figure(fig, output_path)


def plot_phe_fhe_relationship(df: pd.DataFrame, output_path: str | Path) -> Path:
    _require_columns(df, ["PHE", "FHE", "class_label"])
    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    _scatter_by_class(df, axis, "FHE", "PHE")
    axis.set_xlabel("FHE")
    axis.set_ylabel("PHE")
    axis.set_title("PHE and FHE relationship")
    return _save_figure(fig, output_path)


def plot_vocalizations_per_singer_by_voice_type(df: pd.DataFrame, output_path: str | Path) -> Path:
    _require_columns(df, ["singer_id", "voice_type", "class_label"])
    singer_counts = (
        df.groupby(["singer_id", "voice_type", "class_label"], sort=True)
        .size()
        .rename("n_vocalizations")
        .reset_index()
    )

    voice_types = _ordered_voice_types(singer_counts["voice_type"].unique())
    fig, axes = plt.subplots(
        1,
        len(voice_types),
        figsize=(4.2 * len(voice_types), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(voice_types) == 1:
        axes = [axes]

    for axis, voice_type in zip(axes, voice_types):
        voice_counts = (
            singer_counts.loc[singer_counts["voice_type"] == voice_type]
            .sort_values(["class_label", "n_vocalizations", "singer_id"])
            .reset_index(drop=True)
        )
        x_positions = range(len(voice_counts))
        colors = [CLASS_COLORS.get(label, "#4a5568") for label in voice_counts["class_label"]]
        axis.bar(x_positions, voice_counts["n_vocalizations"], color=colors, width=0.85)
        axis.set_title(voice_type)
        axis.set_xlabel("singers")
        axis.set_xticks([])
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("vocalizations")
    fig.suptitle("Vocalizations per singer by voice type")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in CLASS_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    return _save_figure(fig, output_path)


def _ordered_voice_types(values: list[str] | pd.Series | pd.Index) -> list[str]:
    preferred_order = ["soprano", "tenor", "baritone", "bass"]
    present = {str(value) for value in values}
    ordered = [voice_type for voice_type in preferred_order if voice_type in present]
    ordered.extend(sorted(present - set(preferred_order)))
    return ordered


def _correlation_records(
    df: pd.DataFrame,
    feature_cols: list[str],
    scope: str,
    voice_type: str | None = None,
    class_label: str | None = None,
) -> list[dict[str, object]]:
    records = []
    for feature_x, feature_y in combinations(feature_cols, 2):
        pair_df = df[[feature_x, feature_y]].dropna()
        correlation = pair_df[feature_x].corr(pair_df[feature_y]) if len(pair_df) >= 2 else float("nan")
        records.append(
            {
                "scope": scope,
                "voice_type": voice_type,
                "class_label": class_label,
                "feature_x": feature_x,
                "feature_y": feature_y,
                "correlation": correlation,
                "n": len(pair_df),
            }
        )
    return records


def _scatter_by_class(df: pd.DataFrame, axis: plt.Axes, x_col: str, y_col: str) -> None:
    for class_label, class_df in df.groupby("class_label", sort=True):
        axis.scatter(
            class_df[x_col],
            class_df[y_col],
            s=18,
            alpha=0.55,
            label=class_label,
            color=CLASS_COLORS.get(class_label),
            edgecolors="none",
        )
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
