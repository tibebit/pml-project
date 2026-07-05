from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.schema import CORE_FEATURE_COLUMNS
from src.eda import (
    compute_feature_correlations,
    describe_features_by_voice_class,
    plot_feature_distributions,
    plot_phe_fhe_relationship,
    plot_phe_sc_scatter,
    plot_vocalizations_per_singer_by_voice_type,
)
from src.paths import OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR, VOCALIZATION_TABLE_PATH, ensure_project_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe and plot the vocalization-level dataset.")
    parser.add_argument("--input", type=Path, default=VOCALIZATION_TABLE_PATH)
    parser.add_argument("--tables-dir", type=Path, default=OUTPUT_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=OUTPUT_FIGURES_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Vocalization table not found: {args.input}. "
            "Run python3 scripts/02_build_dataset.py first."
        )

    ensure_project_dirs()
    df = pd.read_csv(args.input)

    descriptive = describe_features_by_voice_class(df, CORE_FEATURE_COLUMNS)
    correlations = compute_feature_correlations(df)

    _write_table(descriptive, args.tables_dir / "descriptive_by_voice_class.csv")
    _write_table(correlations, args.tables_dir / "feature_correlations.csv")

    plot_feature_distributions(df, args.figures_dir / "feature_distributions.png", CORE_FEATURE_COLUMNS)
    plot_phe_sc_scatter(df, args.figures_dir / "phe_sc_scatter.png")
    plot_phe_fhe_relationship(df, args.figures_dir / "phe_fhe_relationship.png")
    plot_vocalizations_per_singer_by_voice_type(
        df,
        args.figures_dir / "vocalizations_per_singer_by_voice_type.png",
    )

    print(f"Read {args.input}")
    print(f"Rows: {len(df)}")
    print(f"Singers: {df['singer_id'].nunique()}")
    print(f"Wrote EDA tables to {args.tables_dir}")
    print(f"Wrote EDA figures to {args.figures_dir}")
    return 0


def _write_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
