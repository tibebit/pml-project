# Schema declarations for the raw workbook and processed tables

from __future__ import annotations

from collections.abc import Mapping

VOCALIZATION_TABLE_COLUMNS = [
    "sample_id",
    "source_sheet",
    "raw_voice_code",
    "voice_type",
    "class_label",
    "class_id",
    "raw_person_code",
    "raw_singer_code",
    "singer_id",
    "piece_or_context_code",
    "vowel_code",
    "pitch_code",
    "take",
    "PHE",
    "FHE",
    "SC",
]

CORE_FEATURE_COLUMNS = ["PHE", "FHE", "SC"]

IDENTIFIER_COLUMNS = {
    "sample_id",
    "source_sheet",
    "raw_voice_code",
    "raw_person_code",
    "raw_singer_code",
    "singer_id",
    "piece_or_context_code",
    "vowel_code",
    "pitch_code",
    "take",
}

CLASS_LABEL_TO_ID = {
    "lyric": 0,
    "dramatic": 1,
}

VOICE_CODE_METADATA = {
    "S1": ("soprano", "dramatic"),
    "S2": ("soprano", "lyric"),
    "T1": ("tenor", "dramatic"),
    "T2": ("tenor", "lyric"),
    "B1": ("baritone", "dramatic"),
    "B2": ("baritone", "lyric"),
    "L1": ("bass", "dramatic"),
    "L2": ("bass", "lyric"),
}

CLASS_SHEET_METADATA = {
    "Sop-dram": ("soprano", "dramatic"),
    "Sop-lyr": ("soprano", "lyric"),
    "Ten-dram": ("tenor", "dramatic"),
    "Ten-lyr": ("tenor", "lyric"),
    "Bar-dram": ("baritone", "dramatic"),
    "Bar-lyr": ("baritone", "lyric"),
    "Bass-dram": ("bass", "dramatic"),
    "Bass-lyr": ("bass", "lyric"),
}

ALLE_SHEET_BY_VOICE_TYPE = {
    "soprano": "Sop-Alle",
    "tenor": "Ten-Alle",
    "baritone": "Bar-Alle",
    "bass": "Bass-Alle",
}

_COLUMN_NAME_MAP = {
    "datei": "filename",
    "phe in %": "PHE",
    "fhe hz": "FHE",
    "fhe in hz": "FHE",
    "sc in hz": "SC",
    "s-centroid in hz": "SC",
    "centroid in hz": "SC",
}


def normalize_column_name(column_name: object) -> str:
    name = str(column_name).strip()
    base_name = name.rsplit(".", 1)[0] if name.rsplit(".", 1)[-1].isdigit() else name
    return _COLUMN_NAME_MAP.get(base_name.lower(), name)


def normalize_columns(columns: Mapping[object, object] | list[object]) -> dict[object, str]:
    return {column: normalize_column_name(column) for column in columns}
