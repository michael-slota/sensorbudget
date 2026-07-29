"""Shared schema and source metadata for the occupancy dataset."""

from __future__ import annotations

from pathlib import Path

SOURCE_URL = (
    "https://archive.ics.uci.edu/dataset/357/occupancy%2Bdetection"
)
SOURCE_DOI = "10.24432/C5X01N"
SOURCE_LICENSE = "CC BY 4.0"

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CHECKSUM_PATH = Path("data/source_checksums.json")
DEFAULT_PROCESSED_PATH = Path("data/processed/occupancy.csv")
DEFAULT_MANIFEST_PATH = Path("data/processed/manifest.json")

# The source header omits the first field even though every data row begins
# with an ID. Supplying all names prevents pandas from treating that ID as an
# implicit DataFrame index.
SOURCE_COLUMNS = [
    "source_row_id",
    "date",
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
    "Occupancy",
]

SENSOR_COLUMNS = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
]

OUTPUT_COLUMNS = [*SOURCE_COLUMNS, "source_split"]

# Dictionary order is intentional: it defines the canonical source-file order.
SOURCE_FILES = {
    "train": "datatraining.txt",
    "test_1": "datatest.txt",
    "test_2": "datatest2.txt",
}

EXPECTED_ROW_COUNTS = {
    "train": 8_143,
    "test_1": 2_665,
    "test_2": 9_752,
}
