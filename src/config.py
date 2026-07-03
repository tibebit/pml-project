"""Project-wide constants.

This file is intentionally small and declarative. It is the place to check
which raw files are expected, which feature set defines V1, which voice types
are treated as primary, and which model contract the code is implementing.
Runtime paths live in `src.paths`; statistical code lives in the relevant
dataset/model/evaluation modules.
"""

RAW_ALL_DATA_FILENAME = "Timbre-Nov-21-SciRep-All-Data.xlsx"
RAW_EXAMPLES_FILENAME = "Timbre-Vib-Nov-21-SciRep-Examples-Data.xlsx"

FEATURE_SET_CANDIDATES = {
    "phe_sc": ["PHE", "SC"],
    "fhe_sc": ["FHE", "SC"],
    "phe_fhe_sc": ["PHE", "FHE", "SC"],
}
DEFAULT_FEATURE_SET_NAME = "phe_sc"

MODEL_VERSION = "hierarchical_diagonal_v1"
MODEL_SCOPE = "single_model_indexed_by_voice_type"
COVARIANCE = "diagonal"

PRIMARY_VOICE_TYPES = ("soprano", "tenor", "baritone")
EXPLORATORY_VOICE_TYPES = ("bass",)

RANDOM_SEED = 2026
