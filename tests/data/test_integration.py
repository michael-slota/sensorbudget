from __future__ import annotations

from pathlib import Path

import pytest

from sensorbudget.data.load import (
    combine_source_splits,
    load_source_splits,
)
from sensorbudget.data.schema import EXPECTED_ROW_COUNTS
from sensorbudget.data.validate import validate_source_splits

RAW_DIR = Path("data/raw")
RAW_DATA_AVAILABLE = all(
    (RAW_DIR / filename).is_file()
    for filename in ("datatraining.txt", "datatest.txt", "datatest2.txt")
)


@pytest.mark.skipif(
    not RAW_DATA_AVAILABLE,
    reason="Ignored UCI source files are not available.",
)
def test_local_source_files_satisfy_full_contract() -> None:
    frames = load_source_splits(RAW_DIR)
    summaries = validate_source_splits(frames)
    combined = combine_source_splits(frames)

    assert {split: summary["rows"] for split, summary in summaries.items()} == (
        EXPECTED_ROW_COUNTS
    )
    assert len(combined) == 20_560
    assert combined["source_split"].value_counts().to_dict() == {
        "test_2": 9_752,
        "train": 8_143,
        "test_1": 2_665,
    }
