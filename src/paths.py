"""Centralized filesystem paths for the project.

Keeping paths here prevents scripts and modules from hardcoding local
directories. Raw data, processed CSVs, model artifacts, figures, and tables all
have one canonical location.
"""

from pathlib import Path

from src.config import RAW_ALL_DATA_FILENAME, RAW_EXAMPLES_FILENAME


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_ALL_DATA_PATH = RAW_DATA_DIR / RAW_ALL_DATA_FILENAME
RAW_EXAMPLES_DATA_PATH = RAW_DATA_DIR / RAW_EXAMPLES_FILENAME
VOCALIZATION_TABLE_PATH = PROCESSED_DATA_DIR / "vocalization_table.csv"
SINGER_LEVEL_TABLE_PATH = PROCESSED_DATA_DIR / "singer_level_table.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUT_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUT_REPORTS_DIR = OUTPUTS_DIR / "reports"


def ensure_project_dirs() -> None:
    """Create the project directories that scripts are allowed to write to.

    Scripts call this before saving CSVs, figures, model artifacts, or reports.
    It does not download data and does not clean existing outputs.
    """
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_TABLES_DIR,
        OUTPUT_FIGURES_DIR,
        OUTPUT_MODELS_DIR,
        OUTPUT_REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
