from __future__ import annotations

from pathlib import Path

import pandas as pd

from sensorbudget.data.load import combine_source_splits, load_source_file
from sensorbudget.data.schema import OUTPUT_COLUMNS

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


def test_load_source_file_aligns_omitted_id_header() -> None:
    frame = load_source_file(FIXTURE_DIR / "source_valid.txt", "train")

    assert frame.columns.tolist() == OUTPUT_COLUMNS
    assert frame["source_row_id"].tolist() == [1, 2]
    assert frame["date"].tolist() == [
        pd.Timestamp("2015-02-04 17:51:00"),
        pd.Timestamp("2015-02-04 17:52:00"),
    ]
    assert frame["source_split"].eq("train").all()


def test_combine_source_splits_sorts_time_and_preserves_labels() -> None:
    later = load_source_file(
        FIXTURE_DIR / "source_later.txt",
        "test_1",
    )
    earlier = load_source_file(
        FIXTURE_DIR / "source_earlier.txt",
        "train",
    )

    combined = combine_source_splits({"test_1": later, "train": earlier})

    assert combined["date"].is_monotonic_increasing
    assert combined["source_split"].tolist() == ["train", "test_1"]
