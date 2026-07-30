"""Evaluate occupancy models across physical-sensor cost scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.provenance import validate_source_checksums
from sensorbudget.data.schema import (
    DEFAULT_CHECKSUM_PATH,
    DEFAULT_RAW_DIR,
    SENSOR_COLUMNS,
)
from sensorbudget.data.validate import validate_source_splits
from sensorbudget.modeling.artifacts import (
    save_model_bundle,
    write_json,
    write_table,
)
from sensorbudget.modeling.evaluate import (
    evaluate_fitted_model,
    select_best_models,
    summarize_cross_validation,
    time_series_cross_validate,
)
from sensorbudget.modeling.models import build_baseline_estimators
from sensorbudget.modeling.schema import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_THRESHOLD,
    TARGET_COLUMN,
)

DEFAULT_CONFIG_PATH = Path("configs/sensor_budget.json")
DEFAULT_OUTPUT_DIR = Path("models/sensor_budget")


def load_sensor_budget_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate physical-sensor combinations and cost assumptions."""

    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    sensors = config.get("physical_sensors", {})
    combinations = config.get("feature_sets", [])
    derived = config.get("derived_features", {})

    if not sensors or not combinations:
        raise ValueError(
            "Sensor-budget config requires physical_sensors and feature_sets."
        )

    allowed_features = set(SENSOR_COLUMNS)
    for sensor_name, sensor in sensors.items():
        if float(sensor["cost"]) < 0:
            raise ValueError(f"Sensor {sensor_name!r} has a negative cost.")
        features = sensor.get("features", [])
        if not features or not set(features).issubset(allowed_features):
            raise ValueError(
                f"Sensor {sensor_name!r} has invalid features: {features}."
            )

    seen: set[tuple[str, ...]] = set()
    for combination in combinations:
        key = tuple(combination)
        if not key or len(key) != len(set(key)):
            raise ValueError(f"Invalid sensor combination: {combination}.")
        if not set(key).issubset(sensors):
            raise ValueError(f"Unknown sensor in combination: {combination}.")
        if key in seen:
            raise ValueError(f"Duplicate sensor combination: {combination}.")
        seen.add(key)

    for feature_name, rule in derived.items():
        if feature_name not in allowed_features:
            raise ValueError(f"Unknown derived feature: {feature_name}.")
        if not set(rule.get("requires", [])).issubset(sensors):
            raise ValueError(
                f"Derived feature {feature_name!r} has unknown requirements."
            )
        if float(rule.get("incremental_cost", 0.0)) != 0:
            raise ValueError(
                "Derived features must have zero incremental sensor cost."
            )
    return config


def build_sensor_scenarios(
    config: dict[str, Any],
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Convert configured physical sensors into model features and costs."""

    sensors = config["physical_sensors"]
    derived = config.get("derived_features", {})
    feature_sets: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for combination in config["feature_sets"]:
        name = "__".join(combination)
        features = [
            feature
            for sensor_name in combination
            for feature in sensors[sensor_name]["features"]
        ]
        derived_names = []
        for feature_name, rule in derived.items():
            if set(rule["requires"]).issubset(combination):
                features.append(feature_name)
                derived_names.append(feature_name)

        feature_sets[name] = features
        rows.append(
            {
                "feature_set": name,
                "physical_sensor_count": len(combination),
                "physical_sensors": ", ".join(combination),
                "model_features": ", ".join(features),
                "derived_features": ", ".join(derived_names),
                "relative_cost": float(
                    sum(float(sensors[name]["cost"]) for name in combination)
                ),
                "cost_unit": config["cost_unit"],
            }
        )

    return feature_sets, pd.DataFrame(rows)


def _json_metrics(metrics: dict[str, float | int]) -> dict[str, Any]:
    return {
        key: (
            None
            if isinstance(value, float) and np.isnan(value)
            else value
        )
        for key, value in metrics.items()
    }


def run_sensor_budget(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    n_splits: int = DEFAULT_CV_SPLITS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Select and test a model for every configured sensor combination."""

    validate_source_checksums(raw_dir, checksum_path)
    frames = load_source_splits(raw_dir)
    validate_source_splits(frames)

    config_source = Path(config_path)
    config = load_sensor_budget_config(config_source)
    feature_sets, scenarios = build_sensor_scenarios(config)
    estimators = build_baseline_estimators(random_seed)
    training = frames["train"].sort_values("date").reset_index(drop=True)

    fold_metrics = time_series_cross_validate(
        training,
        estimators,
        feature_sets,
        n_splits=n_splits,
        threshold=threshold,
    )
    cv_summary = summarize_cross_validation(fold_metrics)
    selected = select_best_models(cv_summary)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "sensor_scenarios.csv", scenarios)
    write_table(output / "cv_fold_metrics.csv", fold_metrics)
    write_table(output / "cv_summary.csv", cv_summary)

    selected_rows = []
    heldout_rows = []
    prediction_frames = []
    model_records: dict[str, Any] = {}

    for feature_set, model_name in selected.items():
        feature_columns = feature_sets[feature_set]
        fitted = clone(estimators[model_name])
        with threadpool_limits(limits=1):
            fitted.fit(training[feature_columns], training[TARGET_COLUMN])
        model_path = save_model_bundle(
            output,
            feature_set,
            model_name,
            fitted,
            feature_columns,
            threshold,
        )
        scenario = scenarios.loc[
            scenarios["feature_set"] == feature_set
        ].iloc[0]
        selected_rows.append(
            {
                **scenario.to_dict(),
                "selected_model": model_name,
                "cv_f1_mean": float(
                    cv_summary.loc[
                        (cv_summary["feature_set"] == feature_set)
                        & (cv_summary["model"] == model_name),
                        "f1_mean",
                    ].iloc[0]
                ),
            }
        )
        model_records[feature_set] = {
            "model": model_name,
            "features": feature_columns,
            "relative_cost": float(scenario["relative_cost"]),
            "artifact": model_path.name,
        }

        for split in ("test_1", "test_2"):
            metrics, predictions = evaluate_fitted_model(
                fitted,
                frames[split],
                feature_columns,
                threshold=threshold,
            )
            heldout_rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "split": split,
                    "relative_cost": float(scenario["relative_cost"]),
                    "physical_sensor_count": int(
                        scenario["physical_sensor_count"]
                    ),
                    **metrics,
                }
            )
            predictions.insert(0, "feature_set", feature_set)
            predictions.insert(1, "model", model_name)
            prediction_frames.append(predictions)

    selected_models = pd.DataFrame(selected_rows).sort_values(
        ["relative_cost", "feature_set"]
    )
    heldout_metrics = pd.DataFrame(heldout_rows).sort_values(
        ["split", "f1"],
        ascending=[True, False],
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_table(output / "selected_models.csv", selected_models)
    write_table(output / "heldout_metrics.csv", heldout_metrics)
    write_table(output / "heldout_predictions.csv", predictions)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_source),
        "config_sha256": hashlib.sha256(
            config_source.read_bytes()
        ).hexdigest(),
        "cost_unit": config["cost_unit"],
        "cost_note": config["cost_note"],
        "random_seed": random_seed,
        "cv_splits": n_splits,
        "threshold": threshold,
        "selection_metric": "mean chronological CV F1",
        "training_rows": int(len(training)),
        "scenario_count": int(len(scenarios)),
        "selected_models": model_records,
        "heldout_metrics": [
            {
                key: value
                for key, value in {
                    **row,
                    **_json_metrics(
                        {
                            metric: row[metric]
                            for metric in heldout_metrics.columns
                            if metric
                            not in {
                                "feature_set",
                                "model",
                                "split",
                            }
                        }
                    ),
                }.items()
            }
            for row in heldout_metrics.to_dict(orient="records")
        ],
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json(output / "metadata.json", metadata)
    return {
        "scenarios": scenarios,
        "fold_metrics": fold_metrics,
        "cv_summary": cv_summary,
        "selected_models": selected_models,
        "heldout_metrics": heldout_metrics,
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sensor-budget experiment from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--checksums",
        type=Path,
        default=DEFAULT_CHECKSUM_PATH,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    result = run_sensor_budget(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        config_path=args.config,
        output_dir=args.output_dir,
        n_splits=args.cv_splits,
        random_seed=args.seed,
        threshold=args.threshold,
    )
    print("Selected models by sensor combination:")
    print(result["selected_models"].to_string(index=False))
    print("\nHeld-out metrics:")
    columns = [
        "feature_set",
        "model",
        "split",
        "relative_cost",
        "precision",
        "recall",
        "f1",
    ]
    print(result["heldout_metrics"][columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
