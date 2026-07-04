from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.checks import (
    check_alle_sheets_match_class_sheets,
    check_class_ids,
    check_examples_are_in_all_data,
    check_no_identifier_features,
    check_no_summary_rows,
    check_required_columns_present,
    check_sample_id_metadata_consistency,
    check_unique_sample_ids,
)
from src.dataset.tables import load_vocalizations_from_workbook
from src.paths import (
    OUTPUT_TABLES_DIR,
    RAW_ALL_DATA_PATH,
    RAW_EXAMPLES_DATA_PATH,
    ensure_project_dirs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check extracted vocalizations from raw PML workbooks.")
    parser.add_argument("--raw-all-data", type=Path, default=RAW_ALL_DATA_PATH)
    parser.add_argument("--examples-data", type=Path, default=RAW_EXAMPLES_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_TABLES_DIR / "extracted_vocalization_checks.csv",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    records: list[dict[str, str]] = []

    if not args.raw_all_data.exists():
        records.append(
            {
                "check": "raw_all_data_exists",
                "status": "fail",
                "message": f"Workbook not found: {args.raw_all_data}",
            }
        )
        _write_records(records, args.output)
        return 1

    df = load_vocalizations_from_workbook(args.raw_all_data)

    _run_check(records, "required_columns_present", lambda: check_required_columns_present(df))
    _run_check(records, "no_summary_rows", lambda: check_no_summary_rows(df))
    _run_check(records, "unique_sample_ids", lambda: check_unique_sample_ids(df))
    _run_check(records, "class_id_binary", lambda: check_class_ids(df))
    _run_check(records, "sample_id_metadata_consistency", lambda: check_sample_id_metadata_consistency(df))
    _run_check(records, "no_identifier_features", check_no_identifier_features)
    _run_check(records, "alle_sheets_match_class_sheets", lambda: check_alle_sheets_match_class_sheets(args.raw_all_data))

    if args.examples_data.exists():
        _run_check(records, "examples_are_in_all_data", lambda: check_examples_are_in_all_data(args.examples_data, df))
    else:
        records.append(
            {
                "check": "examples_are_in_all_data",
                "status": "skip",
                "message": f"Examples workbook not found: {args.examples_data}",
            }
        )

    _write_records(records, args.output)
    failed = any(record["status"] == "fail" for record in records)
    return 1 if failed else 0


def _run_check(records: list[dict[str, str]], name: str, check: Callable[[], None]) -> None:
    """Append a pass/fail row for one extraction check."""
    try:
        check()
    except Exception as exc:
        records.append({"check": name, "status": "fail", "message": str(exc)})
    else:
        records.append({"check": name, "status": "pass", "message": ""})


def _write_records(records: list[dict[str, str]], output_path: Path) -> None:
    """Write the extraction check report as a CSV table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
