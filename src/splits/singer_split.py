from __future__ import annotations

import random

import pandas as pd


def make_stratum_column(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ["voice_type", "class_label"])
    return df["voice_type"].astype(str) + "_" + df["class_label"].astype(str)


def make_final_holdout_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    singer_table = _make_singer_table(df)
    # Split by singer, then expand back to all rows
    test_singers = _select_test_singers_by_stratum(singer_table, test_size, seed)
    train_df = df.loc[~df["singer_id"].isin(test_singers)].copy()
    test_df = df.loc[df["singer_id"].isin(test_singers)].copy()
    check_no_singer_leakage(train_df, test_df)
    return train_df, test_df


def check_no_singer_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    _require_columns(train_df, ["singer_id"])
    _require_columns(test_df, ["singer_id"])
    overlap = set(train_df["singer_id"]) & set(test_df["singer_id"])
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"Singer leakage between train and test: {examples}")


def _make_singer_table(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["singer_id", "voice_type", "class_label"])
    singer_table = df[["singer_id", "voice_type", "class_label"]].drop_duplicates()
    # One singer cannot have two labels or two voice types
    duplicated_singers = singer_table["singer_id"].duplicated(keep=False)
    if duplicated_singers.any():
        examples = singer_table.loc[duplicated_singers, "singer_id"].unique().tolist()[:5]
        raise ValueError(f"Inconsistent metadata for singer_id values: {examples}")

    singer_table = singer_table.copy()
    # Stratify by voice_type x class_label so that train/test
    # preserve all the groups used by the model.
    singer_table["stratum"] = make_stratum_column(singer_table)
    return singer_table.sort_values(["stratum", "singer_id"]).reset_index(drop=True)


def _select_test_singers_by_stratum(
    singer_table: pd.DataFrame,
    test_size: float,
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    test_singers: set[str] = set()

    for _, stratum_df in singer_table.groupby("stratum", sort=True):
        singers = sorted(stratum_df["singer_id"])
        rng.shuffle(singers)
        n_singers = len(singers)
        if n_singers == 1:
            n_test = 0
        else:
            # If a group has more than one singer, keep at least one in train.
            n_test = max(1, min(n_singers - 1, round(n_singers * test_size)))
        test_singers.update(singers[:n_test])

    if not test_singers:
        raise ValueError("No test singers selected; increase test_size or provide more singers")
    if len(test_singers) == len(singer_table):
        raise ValueError("All singers selected for test; decrease test_size")
    return test_singers


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
