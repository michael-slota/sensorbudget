"""Shared definitions for baseline occupancy models."""

from __future__ import annotations

from pathlib import Path

from sensorbudget.data.schema import SENSOR_COLUMNS

TARGET_COLUMN = "Occupancy"
DEFAULT_MODEL_DIR = Path("models/baseline")
DEFAULT_THRESHOLD = 0.5
DEFAULT_RANDOM_SEED = 42
DEFAULT_CV_SPLITS = 5

FEATURE_SETS = {
    "all_sensors": list(SENSOR_COLUMNS),
    "no_light": [
        column for column in SENSOR_COLUMNS if column != "Light"
    ],
}

