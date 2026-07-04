from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset.schema import VOICE_CODE_METADATA


# Parse the metadata encoded in workbook sample names
_FILENAME_RE = re.compile(
    r"^(?P<raw_voice_code>[A-Z][0-9])-"
    r"(?P<raw_person_code>[wm][0-9]+)-"
    r"(?P<piece_or_context_code>[^-\s]+)-"
    r"(?P<vowel_code>[^-\s]+)-"
    r"(?P<pitch_code>[^-\s]+)-"
    r"(?P<take>[0-9]+)"
    r"(?:-[^-\s]+)?$"
)

_STATS_PREFIXES = ("durchschnitt", "sd", "test")


def require_workbook(path: str | Path) -> Path:
    workbook_path = Path(path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    return workbook_path


def read_excel_sheets(path: str | Path, sheet_names: list[str]) -> dict[str, pd.DataFrame]:
    workbook_path = require_workbook(path)
    return {sheet_name: pd.read_excel(workbook_path, sheet_name=sheet_name) for sheet_name in sheet_names}


def is_statistic_row(value: Any) -> bool:
    text = "" if value is None else str(value).strip().lower()
    return any(text.startswith(prefix) for prefix in _STATS_PREFIXES)

# Parse a sample filename into voice, class, singer and recording metadata
def parse_filename(filename: Any) -> dict[str, Any]:
    text = "" if filename is None else str(filename).strip()
    if not text:
        raise ValueError("Empty filename")
    if is_statistic_row(text):
        raise ValueError(f"Statistic row is not a sample filename: {text}")

    match = _FILENAME_RE.match(text)
    if match is None:
        raise ValueError(f"Invalid sample filename: {text}")

    parsed = match.groupdict()
    raw_voice_code = parsed["raw_voice_code"]
    if raw_voice_code not in VOICE_CODE_METADATA:
        raise ValueError(f"Unknown raw voice code: {raw_voice_code}")

    voice_type, class_label = VOICE_CODE_METADATA[raw_voice_code]
    raw_person_code = parsed["raw_person_code"]
    
    parsed["voice_type"] = voice_type
    parsed["class_label"] = class_label
    parsed["raw_singer_code"] = f"{raw_voice_code}-{raw_person_code}"
    parsed["singer_id"] = f"{voice_type}_{class_label}_{raw_person_code}"
    parsed["take"] = int(parsed["take"])
    return parsed
