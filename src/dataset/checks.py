# Consistency checks for extracted vocalization data.

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.dataset.schema import (
    ALLE_SHEET_BY_VOICE_TYPE,
    CLASS_SHEET_METADATA,
    IDENTIFIER_COLUMNS,
    VOCALIZATION_TABLE_COLUMNS,
    normalize_columns,
)
from src.dataset.workbook import is_statistic_row, parse_filename, read_excel_sheets


class ExtractedVocalizationError(ValueError):
    """Raised when an extracted table violates the dataset contract."""

    pass


def check_no_summary_rows(df: pd.DataFrame) -> None:
    invalid = df["sample_id"].astype(str).map(is_statistic_row) | df["sample_id"].astype(str).str.strip().eq("")
    if invalid.any():
        examples = df.loc[invalid, "sample_id"].head(5).tolist()
        raise ExtractedVocalizationError(f"Summary/statistic or empty rows found: {examples}")


def check_required_columns_present(df: pd.DataFrame) -> None:
    missing_columns = [column for column in VOCALIZATION_TABLE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ExtractedVocalizationError(f"Missing vocalization table columns: {missing_columns}")

    missing_core = df[VOCALIZATION_TABLE_COLUMNS].isna().any()
    columns_with_missing = missing_core[missing_core].index.tolist()
    if columns_with_missing:
        raise ExtractedVocalizationError(f"Missing values in vocalization table columns: {columns_with_missing}")


def check_unique_sample_ids(df: pd.DataFrame) -> None:
    duplicated = df["sample_id"].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "sample_id"].head(5).tolist()
        raise ExtractedVocalizationError(f"Duplicated sample_id values: {examples}")


def check_sample_id_metadata_consistency(df: pd.DataFrame) -> None:
    for row in df.itertuples(index=False):
        parsed = parse_filename(row.sample_id)
        for column in (
            "raw_voice_code",
            "voice_type",
            "class_label",
            "raw_person_code",
            "raw_singer_code",
            "singer_id",
            "piece_or_context_code",
            "vowel_code",
            "pitch_code",
            "take",
        ):
            if getattr(row, column) != parsed[column]:
                raise ExtractedVocalizationError(
                    f"{row.sample_id} has inconsistent {column}: "
                    f"{getattr(row, column)} != {parsed[column]}"
                )


def check_class_ids(df: pd.DataFrame) -> None:
    observed = set(df["class_id"].unique())
    if observed != {0, 1}:
        raise ExtractedVocalizationError(f"class_id must be {{0, 1}}, found {observed}")


def check_no_identifier_features(feature_cols: Iterable[str] | None = None) -> None:
    candidate_cols = set(feature_cols or [])

    leaked = sorted(candidate_cols & IDENTIFIER_COLUMNS)
    if leaked:
        raise ExtractedVocalizationError(f"Identifier columns cannot be predictive features: {leaked}")


def check_alle_sheets_match_class_sheets(raw_all_data_path: str | Path) -> None:
    sheet_names = list(CLASS_SHEET_METADATA) + list(ALLE_SHEET_BY_VOICE_TYPE.values())
    sheets = read_excel_sheets(raw_all_data_path, sheet_names)

    for voice_type, alle_sheet_name in ALLE_SHEET_BY_VOICE_TYPE.items():
        class_sheet_names = [
            sheet_name
            for sheet_name, metadata in CLASS_SHEET_METADATA.items()
            if metadata[0] == voice_type
        ]
        alle_records = _sheet_sample_records(sheets[alle_sheet_name])
        class_records = set()
        for sheet_name in class_sheet_names:
            class_records.update(_sheet_sample_records(sheets[sheet_name]))

        if alle_records != class_records:
            missing_from_alle = sorted(class_records - alle_records)[:5]
            extra_in_alle = sorted(alle_records - class_records)[:5]
            raise ExtractedVocalizationError(
                f"{alle_sheet_name} does not equal dram/lyr union. "
                f"Missing from Alle: {missing_from_alle}; extra in Alle: {extra_in_alle}"
            )


def check_extracted_vocalizations(
    df: pd.DataFrame,
    raw_all_data_path: str | Path | None = None,
) -> None:
    check_required_columns_present(df)
    check_no_summary_rows(df)
    check_unique_sample_ids(df)
    check_class_ids(df)
    check_sample_id_metadata_consistency(df)
    if raw_all_data_path is not None:
        check_alle_sheets_match_class_sheets(raw_all_data_path)


def _sheet_sample_records(sheet_df: pd.DataFrame) -> set[tuple[str, float, float, float]]:
    df = sheet_df.rename(columns=normalize_columns(list(sheet_df.columns)))
    missing = {"filename", "PHE", "FHE", "SC"} - set(df.columns)
    if missing:
        raise ExtractedVocalizationError(f"Cannot check sheet, missing columns: {sorted(missing)}")

    records: set[tuple[str, float, float, float]] = set()
    for _, row in df.iterrows():
        filename = row["filename"]
        if pd.isna(filename) or is_statistic_row(filename) or not str(filename).strip():
            continue
        parse_filename(filename)
        records.add(
            (
                str(filename).strip(),
                round(float(row["PHE"]), 6),
                round(float(row["FHE"]), 6),
                round(float(row["SC"]), 6),
            )
        )
    return records
