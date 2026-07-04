# Build the official processed dataset tables.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pandas import DataFrame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_FEATURE_COLUMNS
from src.dataset.checks import check_extracted_vocalizations
from src.dataset.tables import (
    build_singer_level_table,
    load_vocalizations_from_workbook,
    summarize_vocalization_table,
    write_singer_level_table,
    write_vocalization_table,
)
from src.paths import (
    OUTPUT_TABLES_DIR,
    RAW_ALL_DATA_PATH,
    SINGER_LEVEL_TABLE_PATH,
    VOCALIZATION_TABLE_PATH,
    ensure_project_dirs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the vocalization-level CSV from the raw workbook.")
    parser.add_argument("--raw-all-data", type=Path, default=RAW_ALL_DATA_PATH)
    parser.add_argument("--output", type=Path, default=VOCALIZATION_TABLE_PATH)
    parser.add_argument("--singer-output", type=Path, default=SINGER_LEVEL_TABLE_PATH)
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=OUTPUT_TABLES_DIR / "dataset_counts.csv",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    df = load_vocalizations_from_workbook(args.raw_all_data)
    check_extracted_vocalizations(
        df,
        raw_all_data_path=args.raw_all_data,
    )
    feature_cols = list(MODEL_FEATURE_COLUMNS)
    singer_df = build_singer_level_table(df, feature_cols)

    write_vocalization_table(df, args.output)
    write_singer_level_table(singer_df, args.singer_output)
    _write_counts(df, args.counts_output)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.singer_output}")
    print(f"Rows: {len(df)}")
    print(f"Singers: {df['singer_id'].nunique()}")
    return 0


def _write_counts(df: DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summarize_vocalization_table(df).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
