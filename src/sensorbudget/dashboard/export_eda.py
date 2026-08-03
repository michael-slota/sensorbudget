"""Export aggregated EDA evidence for the static GitHub Pages dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sensorbudget.data.load import load_occupancy_data, load_source_splits
from sensorbudget.data.schema import DEFAULT_RAW_DIR, SENSOR_COLUMNS, SOURCE_FILES

DEFAULT_OUTPUT_PATH = Path("site/data/eda.json")
DISPLAY_NAMES = {"train": "Training", "test_1": "Test 1", "test_2": "Test 2"}


def build_eda_payload(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, object]:
    """Build the aggregate values displayed by the public EDA dashboard."""

    splits = load_source_splits(raw_dir)
    data = load_occupancy_data(raw_dir)

    split_summary = []
    for split_name in SOURCE_FILES:
        frame = splits[split_name]
        occupied_rows = int(frame["Occupancy"].sum())
        split_summary.append(
            {
                "split": split_name,
                "label": DISPLAY_NAMES[split_name],
                "rows": int(len(frame)),
                "occupied_rows": occupied_rows,
                "unoccupied_rows": int(len(frame) - occupied_rows),
                "occupancy_rate": float(frame["Occupancy"].mean()),
                "period_start": frame["date"].min().isoformat(),
                "period_end": frame["date"].max().isoformat(),
            }
        )

    hourly = (
        data.assign(hour=data["date"].dt.hour)
        .groupby(["source_split", "hour"], observed=True)["Occupancy"]
        .agg(occupancy_rate="mean", observations="size")
        .reset_index()
    )
    hourly_rows = [
        {
            "split": str(row.source_split),
            "label": DISPLAY_NAMES[str(row.source_split)],
            "hour": int(row.hour),
            "occupancy_rate": float(row.occupancy_rate),
            "observations": int(row.observations),
        }
        for row in hourly.itertuples(index=False)
    ]

    correlations = data[[*SENSOR_COLUMNS, "Occupancy"]].corr()["Occupancy"]
    correlation_rows = [
        {"sensor": sensor, "correlation": float(correlations[sensor])}
        for sensor in sorted(
            SENSOR_COLUMNS, key=lambda column: abs(correlations[column]), reverse=True
        )
    ]

    light_rows = []
    for split_name in SOURCE_FILES:
        frame = splits[split_name]
        occupied = frame["Occupancy"].eq(1)
        lit = frame["Light"].gt(0)
        occupied_total = int(occupied.sum())
        unoccupied_total = int((~occupied).sum())
        occupied_dark = int((occupied & ~lit).sum())
        unoccupied_lit = int((~occupied & lit).sum())
        light_rows.append(
            {
                "split": split_name,
                "label": DISPLAY_NAMES[split_name],
                "occupied_rows": occupied_total,
                "occupied_while_dark": occupied_dark,
                "occupied_while_dark_rate": occupied_dark / occupied_total,
                "unoccupied_rows": unoccupied_total,
                "unoccupied_while_lit": unoccupied_lit,
                "unoccupied_while_lit_rate": unoccupied_lit / unoccupied_total,
            }
        )

    train = splits["train"]
    drift_rows = []
    for split_name in ("test_1", "test_2"):
        test = splits[split_name]
        for sensor in SENSOR_COLUMNS:
            pooled_std = np.sqrt(
                (train[sensor].var(ddof=1) + test[sensor].var(ddof=1)) / 2
            )
            drift_rows.append(
                {
                    "split": split_name,
                    "label": DISPLAY_NAMES[split_name],
                    "sensor": sensor,
                    "standardized_mean_difference": float(
                        (test[sensor].mean() - train[sensor].mean()) / pooled_std
                    ),
                }
            )

    return {
        "metadata": {
            "title": "Time-aware exploratory analysis",
            "total_rows": int(len(data)),
            "room_count": 1,
            "sampling_interval_seconds": 60,
            "source": "UCI Occupancy Detection",
            "model_card": (
                "https://github.com/michael-slota/sensorbudget/"
                "blob/main/reports/model_card.md"
            ),
        },
        "split_summary": split_summary,
        "hourly_occupancy": hourly_rows,
        "target_correlations": correlation_rows,
        "light_exceptions": light_rows,
        "standardized_drift": drift_rows,
    }


def export_eda_dashboard_data(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the aggregate EDA payload as deterministic formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_eda_payload(raw_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Parse command-line arguments and export the EDA dashboard data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output = export_eda_dashboard_data(args.raw_dir, args.output)
    print(f"Wrote EDA dashboard data to {output}")


if __name__ == "__main__":
    main()
