"""Data-contract validation for the UCI occupancy dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.schema import (
    DEFAULT_RAW_DIR,
    EXPECTED_ROW_COUNTS,
    OUTPUT_COLUMNS,
    SENSOR_COLUMNS,
    SOURCE_FILES,
)


class DataValidationError(ValueError):
    """Raised when source data violates the expected contract."""


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_source_frame(
    frame: pd.DataFrame,
    split_name: str,
    *,
    expected_rows: int | None = None,
    allowed_interval_seconds: tuple[float, float] = (50.0, 70.0),
) -> dict[str, Any]:
    """Validate one supplied period and return a serializable quality summary."""

    errors: list[str] = []
    _require(
        frame.columns.tolist() == OUTPUT_COLUMNS,
        f"{split_name}: columns do not match the expected schema.",
        errors,
    )

    if frame.columns.tolist() != OUTPUT_COLUMNS:
        raise DataValidationError("\n".join(errors))

    _require(len(frame) > 0, f"{split_name}: file contains no rows.", errors)
    if expected_rows is not None:
        _require(
            len(frame) == expected_rows,
            f"{split_name}: expected {expected_rows:,} rows, found {len(frame):,}.",
            errors,
        )

    _require(
        not frame.isna().any().any(),
        f"{split_name}: missing values were found.",
        errors,
    )
    _require(
        pd.api.types.is_datetime64_any_dtype(frame["date"]),
        f"{split_name}: date is not a datetime column.",
        errors,
    )
    _require(
        pd.api.types.is_integer_dtype(frame["source_row_id"]),
        f"{split_name}: source_row_id is not an integer column.",
        errors,
    )
    _require(
        frame["source_row_id"].is_unique,
        f"{split_name}: source_row_id contains duplicates.",
        errors,
    )
    _require(
        frame["date"].is_unique,
        f"{split_name}: duplicate timestamps were found.",
        errors,
    )
    _require(
        frame["date"].is_monotonic_increasing,
        f"{split_name}: timestamps are not chronological.",
        errors,
    )
    _require(
        set(frame["Occupancy"].unique()).issubset({0, 1}),
        f"{split_name}: Occupancy contains values other than 0 and 1.",
        errors,
    )
    _require(
        frame["source_split"].eq(split_name).all(),
        f"{split_name}: source_split contains an incorrect label.",
        errors,
    )

    sensor_values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
    _require(
        bool(np.isfinite(sensor_values).all()),
        f"{split_name}: sensor columns contain infinite values.",
        errors,
    )
    _require(
        bool((frame[SENSOR_COLUMNS] >= 0).all().all()),
        f"{split_name}: sensor columns contain negative values.",
        errors,
    )

    intervals = frame["date"].diff().dt.total_seconds().dropna()
    if not intervals.empty:
        minimum, maximum = allowed_interval_seconds
        _require(
            bool(intervals.between(minimum, maximum).all()),
            (
                f"{split_name}: sampling intervals fall outside "
                f"{minimum:g}-{maximum:g} seconds."
            ),
            errors,
        )

    if errors:
        raise DataValidationError("\n".join(errors))

    return {
        "split": split_name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "start": frame["date"].min().isoformat(),
        "end": frame["date"].max().isoformat(),
        "occupied_rows": int(frame["Occupancy"].sum()),
        "occupied_rate": float(frame["Occupancy"].mean()),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame["date"].duplicated().sum()),
        "median_interval_seconds": float(intervals.median()),
        "minimum_interval_seconds": float(intervals.min()),
        "maximum_interval_seconds": float(intervals.max()),
    }


def validate_source_splits(
    frames: Mapping[str, pd.DataFrame],
    *,
    expected_row_counts: Mapping[str, int] = EXPECTED_ROW_COUNTS,
    required_splits: Sequence[str] = tuple(SOURCE_FILES),
) -> dict[str, dict[str, Any]]:
    """Validate the complete set of supplied periods."""

    missing_splits = [split for split in required_splits if split not in frames]
    unexpected_splits = [split for split in frames if split not in required_splits]
    if missing_splits or unexpected_splits:
        details = []
        if missing_splits:
            details.append(f"missing splits: {', '.join(missing_splits)}")
        if unexpected_splits:
            details.append(f"unexpected splits: {', '.join(unexpected_splits)}")
        raise DataValidationError("; ".join(details))

    summaries = {
        split: validate_source_frame(
            frames[split],
            split,
            expected_rows=expected_row_counts.get(split),
        )
        for split in required_splits
    }

    total_rows = sum(summary["rows"] for summary in summaries.values())
    if total_rows != sum(expected_row_counts.values()):
        raise DataValidationError(
            f"Expected {sum(expected_row_counts.values()):,} total rows, "
            f"found {total_rows:,}."
        )
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """Validate local raw files and print their quality summary as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing the three UCI source files.",
    )
    args = parser.parse_args(argv)

    summaries = validate_source_splits(load_source_splits(args.raw_dir))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

