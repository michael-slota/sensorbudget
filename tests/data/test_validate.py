from __future__ import annotations

from pathlib import Path

import pytest

from sensorbudget.data.load import load_source_file
from sensorbudget.data.validate import (
    DataValidationError,
    validate_source_frame,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


def test_valid_source_frame_returns_quality_summary() -> None:
    frame = load_source_file(
        FIXTURE_DIR / "source_valid.txt",
        "train",
    )

    summary = validate_source_frame(frame, "train", expected_rows=2)

    assert summary["rows"] == 2
    assert summary["occupied_rows"] == 1
    assert summary["median_interval_seconds"] == 60


def test_validation_rejects_non_binary_target() -> None:
    frame = load_source_file(
        FIXTURE_DIR / "source_invalid_target.txt",
        "train",
    )

    with pytest.raises(DataValidationError, match="other than 0 and 1"):
        validate_source_frame(frame, "train", expected_rows=1)


def test_validation_rejects_duplicate_timestamps() -> None:
    frame = load_source_file(
        FIXTURE_DIR / "source_duplicate_time.txt",
        "train",
    )

    with pytest.raises(DataValidationError, match="duplicate timestamps"):
        validate_source_frame(frame, "train", expected_rows=2)
