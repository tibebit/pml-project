from pathlib import Path

import pandas as pd
import pytest

from src.dataset.checks import check_extracted_vocalizations
from src.dataset.schema import VOCALIZATION_TABLE_COLUMNS
from src.dataset.tables import build_singer_level_table, load_vocalizations_from_workbook
from src.paths import RAW_ALL_DATA_PATH


def test_load_vocalizations_from_workbook_filters_stats_and_adds_metadata(tmp_path):
    workbook_path = tmp_path / "raw.xlsx"
    _write_minimal_workbook(workbook_path)

    df = load_vocalizations_from_workbook(workbook_path)

    assert len(df) == 8
    assert list(df.columns) == VOCALIZATION_TABLE_COLUMNS
    assert not df["sample_id"].str.startswith(("Durchschnitt", "SD", "Test")).any()

    row = df.loc[df["sample_id"] == "S1-w001-Uebu-A-a1-1"].iloc[0]
    assert row["sample_id"] == "S1-w001-Uebu-A-a1-1"
    assert row["source_sheet"] == "Sop-dram"
    assert row["singer_id"] == "soprano_dramatic_w001"
    assert row["class_id"] == 1
    assert row["PHE"] == 24.1
    assert row["FHE"] == 2831
    assert row["SC"] == 2976


def test_load_vocalizations_from_workbook_real_workbook_if_available():
    if not RAW_ALL_DATA_PATH.exists():
        pytest.skip(f"Raw workbook not found at {RAW_ALL_DATA_PATH}")

    df = load_vocalizations_from_workbook(RAW_ALL_DATA_PATH)

    assert not df.empty
    assert df["singer_id"].nunique() > 0
    check_extracted_vocalizations(df, raw_all_data_path=RAW_ALL_DATA_PATH)


def test_build_singer_level_table_returns_one_row_per_singer():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "singer_id": ["s1", "s1", "s2"],
            "voice_type": ["soprano", "soprano", "tenor"],
            "class_label": ["dramatic", "dramatic", "lyric"],
            "class_id": [1, 1, 0],
            "PHE": [10.0, 14.0, 20.0],
            "SC": [1000.0, 1200.0, 2000.0],
        }
    )

    singer_df = build_singer_level_table(df, ["PHE", "SC"])

    assert len(singer_df) == df["singer_id"].nunique()
    assert set(singer_df["singer_id"]) == {"s1", "s2"}
    assert "n_vocalizations" in singer_df.columns
    assert "mean_PHE" in singer_df.columns
    assert "sd_SC" in singer_df.columns


def _write_minimal_workbook(path: Path) -> None:
    sheet_rows = {
        "Sop-dram": "S1-w001-Uebu-A-a1-1",
        "Sop-lyr": "S2-w002-Adin-A-a2-1",
        "Ten-dram": "T1-m001-Cani-A-a1-1",
        "Ten-lyr": "T2-m002-Chap-A-a-1",
        "Bar-dram": "B1-m001-Amfo-A-b-1",
        "Bar-lyr": "B2-m002-Belc-A-a1-1",
        "Bass-dram": "L1-m001-Gurn-A-as-1",
        "Bass-lyr": "L2-m002-Dala-A-b-1",
    }
    with pd.ExcelWriter(path) as writer:
        for sheet_name, filename in sheet_rows.items():
            pd.DataFrame(
                {
                    "Datei": [filename, "Durchschnitt aus Matlab", "SD", "Test Mittelwert"],
                    "PHE in %": [24.1, 0.0, 0.0, 0.0],
                    "FHE in Hz": [2831, 0.0, 0.0, 0.0],
                    "S-Centroid in Hz": [2976, 0.0, 0.0, 0.0],
                }
            ).to_excel(writer, sheet_name=sheet_name, index=False)
