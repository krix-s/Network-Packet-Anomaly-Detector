"""Project paths and dataset column conventions."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "NF-CSE-CIC-IDS2018-V2.parquet"
BINARY_TARGET = "Label"
MULTICLASS_TARGET = "Attack"
TARGET_COLUMNS = (BINARY_TARGET, MULTICLASS_TARGET)

# Exclusions for the first safe tree-model baseline. The rate fields are kept
# in the source and should be revisited in an explicit comparison.
INITIAL_EXCLUDED_FEATURES = (
    "FTP_COMMAND_RET_CODE",  # constant zero in the complete dataset audit
    "SRC_TO_DST_SECOND_BYTES",  # finite, but reaches 6.6e304
    "DST_TO_SRC_SECOND_BYTES",  # finite, but reaches 3.2e73
)
