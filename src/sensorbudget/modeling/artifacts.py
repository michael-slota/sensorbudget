"""Persistence helpers for reproducible model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator


def save_model_bundle(
    output_dir: Path | str,
    feature_set: str,
    model_name: str,
    estimator: BaseEstimator,
    feature_columns: list[str],
    threshold: float,
) -> Path:
    """Serialize a fitted estimator together with its prediction contract."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{feature_set}__{model_name}.joblib"
    joblib.dump(
        {
            "feature_set": feature_set,
            "model_name": model_name,
            "feature_columns": feature_columns,
            "threshold": threshold,
            "estimator": estimator,
        },
        path,
    )
    return path


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Write human-readable JSON, including NumPy-compatible values."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def write_table(path: Path | str, frame: pd.DataFrame) -> None:
    """Write a DataFrame as a reproducible CSV artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, date_format="%Y-%m-%dT%H:%M:%S")

