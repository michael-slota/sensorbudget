"""Load the supplied UCI occupancy files without altering raw data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from sensorbudget.data.schema import (
    DEFAULT_RAW_DIR,
    OUTPUT_COLUMNS,
    SOURCE_COLUMNS,
    SOURCE_FILES,
)


def load_source_file(path: Path | str, split_name: str) -> pd.DataFrame:
    """Load one source file using the dataset's explicit eight-column schema.

    The original header lists seven columns while every data row contains an
    additional leading row ID. ``header=0`` skips that incomplete header and
    ``names=SOURCE_COLUMNS`` assigns all eight fields deterministically.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Occupancy source file not found: {source_path}")

    frame = pd.read_csv(
        source_path,
        header=0,
        names=SOURCE_COLUMNS,
        parse_dates=["date"],
    )
    frame["source_split"] = split_name

    # Keep a stable output order even if pandas changes internal inference.
    return frame.loc[:, OUTPUT_COLUMNS]


def load_source_splits(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    source_files: Mapping[str, str] = SOURCE_FILES,
) -> dict[str, pd.DataFrame]:
    """Load every supplied period while keeping each split separate."""

    raw_path = Path(raw_dir)
    return {
        split_name: load_source_file(raw_path / filename, split_name)
        for split_name, filename in source_files.items()
    }


def combine_source_splits(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine source periods for storage while retaining split identity.

    Sorting by timestamp gives a chronological descriptive dataset. Evaluation
    code must still filter on ``source_split`` rather than resplitting rows.
    """

    if not frames:
        raise ValueError("At least one source split is required.")

    combined = pd.concat(frames.values(), ignore_index=True)
    return (
        combined.loc[:, OUTPUT_COLUMNS]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


def load_occupancy_data(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Load and chronologically combine all canonical source files."""

    return combine_source_splits(load_source_splits(raw_dir))

