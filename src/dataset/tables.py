# Build processed vocalization-level and singer-level tables.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.checks import check_no_identifier_features
from src.dataset.schema import (
    CLASS_LABEL_TO_ID,
    CLASS_SHEET_METADATA,
    CORE_FEATURE_COLUMNS,
    VOCALIZATION_TABLE_COLUMNS,
    normalize_columns,
)
from src.dataset.workbook import is_statistic_row, parse_filename, read_excel_sheets
from src.paths import RAW_ALL_DATA_PATH, SINGER_LEVEL_TABLE_PATH, VOCALIZATION_TABLE_PATH


SINGER_METADATA_COLUMNS = ["singer_id", "voice_type", "class_label", "class_id"]


def make_singer_level_feature_columns(feature_cols: list[str]) -> list[str]:
    return (
        [f"mean_{column}" for column in feature_cols]
        + [f"sd_{column}" for column in feature_cols]
    )


def load_vocalizations_from_workbook(raw_all_data_path: str | Path = RAW_ALL_DATA_PATH) -> pd.DataFrame:
    sheets = read_excel_sheets(raw_all_data_path, list(CLASS_SHEET_METADATA))
    frames = [
        _extract_vocalizations_from_class_sheet(sheet_name, sheet_df)
        for sheet_name, sheet_df in sheets.items()
    ]
    vocalizations = pd.concat(frames, ignore_index=True)
    return vocalizations[VOCALIZATION_TABLE_COLUMNS]


def build_singer_level_table(vocalization_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    check_no_identifier_features(feature_cols)
    _require_columns(vocalization_df, SINGER_METADATA_COLUMNS + feature_cols)
    _check_singer_metadata_consistency(vocalization_df)

    grouped = vocalization_df.groupby("singer_id", sort=True)
    metadata = grouped[SINGER_METADATA_COLUMNS[1:]].first()
    counts = grouped.size().rename("n_vocalizations")
    means = grouped[feature_cols].mean().rename(columns={column: f"mean_{column}" for column in feature_cols})
    sds = (
        grouped[feature_cols]
        .std(ddof=1)
        .fillna(0.0)
        .rename(columns={column: f"sd_{column}" for column in feature_cols})
    )

    table = pd.concat([metadata, counts, means, sds], axis=1).reset_index()

    ordered_columns = (
        SINGER_METADATA_COLUMNS
        + ["n_vocalizations"]
        + make_singer_level_feature_columns(feature_cols)
    )
    return table[ordered_columns]


def write_vocalization_table(
    df: pd.DataFrame,
    output_path: str | Path = VOCALIZATION_TABLE_PATH,
) -> Path:
    return _write_csv(df, output_path)


def write_singer_level_table(
    df: pd.DataFrame,
    output_path: str | Path = SINGER_LEVEL_TABLE_PATH,
) -> Path:
    return _write_csv(df, output_path)


def summarize_vocalization_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["voice_type", "class_label"], as_index=False)
        .agg(n_vocalizations=("sample_id", "size"), n_singers=("singer_id", "nunique"))
        .sort_values(["voice_type", "class_label"])
    )
    return counts


def _extract_vocalizations_from_class_sheet(sheet_name: str, sheet_df: pd.DataFrame) -> pd.DataFrame:
    expected_voice_type, expected_class_label = CLASS_SHEET_METADATA[sheet_name]
    df = sheet_df.rename(columns=normalize_columns(list(sheet_df.columns)))
    _require_sheet_columns(sheet_name, df)

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        raw_sample_code = row["filename"]
        if pd.isna(raw_sample_code) or is_statistic_row(raw_sample_code) or not str(raw_sample_code).strip():
            continue

        parsed = parse_filename(raw_sample_code)
        if parsed["voice_type"] != expected_voice_type or parsed["class_label"] != expected_class_label:
            raise ValueError(
                f"{raw_sample_code} metadata does not match sheet {sheet_name}: "
                f"{parsed['voice_type']}/{parsed['class_label']}"
            )

        vocalization_row = {
            "sample_id": str(raw_sample_code).strip(),
            "source_sheet": sheet_name,
            "class_id": CLASS_LABEL_TO_ID[parsed["class_label"]],
        }
        vocalization_row.update(parsed)
        for column in CORE_FEATURE_COLUMNS:
            vocalization_row[column] = pd.to_numeric(row[column], errors="coerce")
        rows.append(vocalization_row)

    return pd.DataFrame(rows, columns=VOCALIZATION_TABLE_COLUMNS)


def _check_singer_metadata_consistency(df: pd.DataFrame) -> None:
    inconsistent = (
        df.groupby("singer_id")[["voice_type", "class_label", "class_id"]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if inconsistent.any():
        examples = inconsistent[inconsistent].index.tolist()[:5]
        raise ValueError(f"Inconsistent singer metadata for singer_id values: {examples}")


def _require_sheet_columns(sheet_name: str, df: pd.DataFrame) -> None:
    required = {"filename", "PHE", "FHE", "SC"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Sheet {sheet_name} is missing required columns: {missing}")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _write_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
