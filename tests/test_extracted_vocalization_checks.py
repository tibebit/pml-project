import pytest

from src.dataset.checks import (
    ExtractedVocalizationError,
    check_alle_sheets_match_class_sheets,
    check_no_identifier_features,
    check_unique_sample_ids,
)
from src.dataset.tables import load_vocalizations_from_workbook
from src.paths import RAW_ALL_DATA_PATH


def test_extracted_vocalization_checks_real_workbook_if_available():
    if not RAW_ALL_DATA_PATH.exists():
        pytest.skip(f"Raw workbook not found at {RAW_ALL_DATA_PATH}")

    df = load_vocalizations_from_workbook(RAW_ALL_DATA_PATH)

    check_unique_sample_ids(df)
    check_alle_sheets_match_class_sheets(RAW_ALL_DATA_PATH)


def test_check_no_identifier_features_rejects_identifier_features():
    with pytest.raises(ExtractedVocalizationError):
        check_no_identifier_features(["PHE", "SC", "singer_id"])
