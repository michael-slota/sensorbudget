"""Evaluate validation-selected sensor configurations under sensor faults."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.provenance import validate_source_checksums
from sensorbudget.data.schema import (
    DEFAULT_CHECKSUM_PATH,
    DEFAULT_RAW_DIR,
)
from sensorbudget.data.validate import validate_source_splits
from sensorbudget.modeling.artifacts import write_json, write_table
from sensorbudget.modeling.evaluate import evaluate_fitted_model
from sensorbudget.modeling.schema import TARGET_COLUMN

DEFAULT_CONFIG_PATH = Path("configs/robustness.json")
DEFAULT_SENSOR_BUDGET_DIR = Path("models/sensor_budget")
DEFAULT_OUTPUT_DIR = Path("models/robustness")


def load_robustness_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the robustness experiment definition."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not config.get("comparison_feature_sets"):
        raise ValueError("Robustness config requires comparison_feature_sets.")
    if not config.get("heldout_splits"):
        raise ValueError("Robustness config requires heldout_splits.")
    if int(config.get("random_repeats", 0)) < 1:
        raise ValueError("random_repeats must be at least one.")

    for key in (
        "random_missing_rates",
        "gaussian_noise_std_fractions",
    ):
        values = [float(value) for value in config.get(key, [])]
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"{key} must contain positive values.")
    if any(
        not 0 < float(rate) < 1
        for rate in config["random_missing_rates"]
    ):
        raise ValueError("random_missing_rates must be between zero and one.")

    quantiles = config.get("stuck_quantiles", {})
    if not quantiles or any(
        not 0 <= float(value) <= 1 for value in quantiles.values()
    ):
        raise ValueError("stuck_quantiles must be between zero and one.")
    if not config.get("drift_std_fractions"):
        raise ValueError("drift_std_fractions cannot be empty.")
    return config


def stable_random_seed(
    base_seed: int,
    *parts: object,
) -> int:
    """Derive a reproducible NumPy seed independent of Python hash order."""

    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def apply_random_missingness(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    rate: float,
    medians: Mapping[str, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    """Randomly remove feature cells and apply training-median imputation."""

    modified = frame.copy()
    mask = rng.random((len(modified), len(feature_columns))) < rate
    values = modified[feature_columns].mask(mask)
    modified.loc[:, feature_columns] = values.fillna(medians)
    return modified, int(mask.sum())


def apply_gaussian_noise(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    std_fraction: float,
    standard_deviations: Mapping[str, float],
    observed_minimums: Mapping[str, float],
    observed_maximums: Mapping[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add bounded Gaussian measurement noise scaled by training spread."""

    modified = frame.copy()
    for feature in feature_columns:
        noise = rng.normal(
            loc=0.0,
            scale=float(standard_deviations[feature]) * std_fraction,
            size=len(modified),
        )
        modified[feature] = (
            modified[feature] + noise
        ).clip(
            lower=float(observed_minimums[feature]),
            upper=float(observed_maximums[feature]),
        )
    return modified


def apply_complete_loss(
    frame: pd.DataFrame,
    sensor: str,
    *,
    median: float,
) -> pd.DataFrame:
    """Replace a completely unavailable sensor with its training median."""

    modified = frame.copy()
    modified[sensor] = median
    return modified


def apply_stuck_sensor(
    frame: pd.DataFrame,
    sensor: str,
    *,
    stuck_value: float,
) -> pd.DataFrame:
    """Freeze one sensor at a fixed training-derived value."""

    modified = frame.copy()
    modified[sensor] = stuck_value
    return modified


def apply_gradual_drift(
    frame: pd.DataFrame,
    sensor: str,
    *,
    final_std_fraction: float,
    training_standard_deviation: float,
) -> pd.DataFrame:
    """Apply a linear offset from zero to a final fraction of training spread."""

    modified = frame.copy()
    offsets = np.linspace(
        0.0,
        final_std_fraction * training_standard_deviation,
        num=len(modified),
    )
    modified[sensor] = modified[sensor].to_numpy() + offsets
    return modified


def apply_light_policy_failure(
    frame: pd.DataFrame,
    *,
    mode: str,
    occupied_light_reference: float,
    unoccupied_light_reference: float,
) -> tuple[pd.DataFrame, int]:
    """Create diagnostic occupied-dark or unoccupied-lit counterfactual rows."""

    modified = frame.copy()
    if mode == "unoccupied_lit":
        affected = modified[TARGET_COLUMN] == 0
        modified.loc[affected, "Light"] = occupied_light_reference
    elif mode == "occupied_dark":
        affected = modified[TARGET_COLUMN] == 1
        modified.loc[affected, "Light"] = unoccupied_light_reference
    else:
        raise ValueError(f"Unknown Light policy failure mode: {mode}.")
    return modified, int(affected.sum())


def _training_statistics(
    training: pd.DataFrame,
    features: set[str],
    stuck_quantiles: Mapping[str, float],
) -> dict[str, Any]:
    ordered_features = sorted(features)
    return {
        "median": training[ordered_features].median().to_dict(),
        "std": training[ordered_features].std().to_dict(),
        "minimum": training[ordered_features].min().to_dict(),
        "maximum": training[ordered_features].max().to_dict(),
        "stuck": {
            label: training[ordered_features]
            .quantile(float(quantile))
            .to_dict()
            for label, quantile in stuck_quantiles.items()
        },
        "occupied_light_reference": float(
            training.loc[training[TARGET_COLUMN] == 1, "Light"].median()
        ),
        "unoccupied_light_reference": float(
            training.loc[training[TARGET_COLUMN] == 0, "Light"].median()
        ),
    }


def _evaluate(
    estimator: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> dict[str, float | int]:
    metrics, _ = evaluate_fitted_model(
        estimator,
        frame,
        feature_columns,
        threshold=threshold,
    )
    return metrics


def run_robustness_evaluation(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    sensor_budget_dir: Path | str = DEFAULT_SENSOR_BUDGET_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run all configured robustness interventions on held-out periods."""

    validate_source_checksums(raw_dir, checksum_path)
    frames = load_source_splits(raw_dir)
    validate_source_splits(frames)

    config_source = Path(config_path)
    config = load_robustness_config(config_source)
    budget_dir = Path(sensor_budget_dir)
    selected_path = budget_dir / "selected_models.csv"
    selected = pd.read_csv(selected_path).set_index("feature_set")

    bundles: dict[str, dict[str, Any]] = {}
    all_features: set[str] = set()
    for feature_set in config["comparison_feature_sets"]:
        if feature_set not in selected.index:
            raise ValueError(
                f"Unknown comparison feature set: {feature_set!r}."
            )
        model_name = str(selected.loc[feature_set, "selected_model"])
        bundle_path = budget_dir / f"{feature_set}__{model_name}.joblib"
        bundle = joblib.load(bundle_path)
        bundles[feature_set] = bundle
        all_features.update(bundle["feature_columns"])

    training = frames["train"].sort_values("date").reset_index(drop=True)
    statistics = _training_statistics(
        training,
        all_features,
        config["stuck_quantiles"],
    )
    rows: list[dict[str, Any]] = []
    base_seed = int(config["random_seed"])

    def record(
        *,
        feature_set: str,
        model_name: str,
        split: str,
        scenario_group: str,
        scenario: str,
        severity: float | str,
        sensor: str,
        repeat: int,
        affected_values: int,
        metrics: dict[str, float | int],
    ) -> None:
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "split": split,
                "scenario_group": scenario_group,
                "scenario": scenario,
                "severity": severity,
                "sensor": sensor,
                "repeat": repeat,
                "affected_values": affected_values,
                **metrics,
            }
        )

    for feature_set, bundle in bundles.items():
        estimator = bundle["estimator"]
        feature_columns = list(bundle["feature_columns"])
        threshold = float(bundle["threshold"])
        model_name = str(bundle["model_name"])

        for split in config["heldout_splits"]:
            evaluation = frames[split].sort_values("date").reset_index(
                drop=True
            )
            record(
                feature_set=feature_set,
                model_name=model_name,
                split=split,
                scenario_group="baseline",
                scenario="baseline",
                severity=0.0,
                sensor="all",
                repeat=0,
                affected_values=0,
                metrics=_evaluate(
                    estimator,
                    evaluation,
                    feature_columns,
                    threshold,
                ),
            )

            if "Light" in feature_columns:
                for mode in ("unoccupied_lit", "occupied_dark"):
                    modified, affected = apply_light_policy_failure(
                        evaluation,
                        mode=mode,
                        occupied_light_reference=statistics[
                            "occupied_light_reference"
                        ],
                        unoccupied_light_reference=statistics[
                            "unoccupied_light_reference"
                        ],
                    )
                    record(
                        feature_set=feature_set,
                        model_name=model_name,
                        split=split,
                        scenario_group="light_policy",
                        scenario=mode,
                        severity=1.0,
                        sensor="Light",
                        repeat=0,
                        affected_values=affected,
                        metrics=_evaluate(
                            estimator,
                            modified,
                            feature_columns,
                            threshold,
                        ),
                    )

            for rate in config["random_missing_rates"]:
                for repeat in range(int(config["random_repeats"])):
                    seed = stable_random_seed(
                        base_seed,
                        feature_set,
                        split,
                        "missing",
                        rate,
                        repeat,
                    )
                    modified, affected = apply_random_missingness(
                        evaluation,
                        feature_columns,
                        rate=float(rate),
                        medians=statistics["median"],
                        rng=np.random.default_rng(seed),
                    )
                    record(
                        feature_set=feature_set,
                        model_name=model_name,
                        split=split,
                        scenario_group="random_missing",
                        scenario="random_missing",
                        severity=float(rate),
                        sensor="all",
                        repeat=repeat + 1,
                        affected_values=affected,
                        metrics=_evaluate(
                            estimator,
                            modified,
                            feature_columns,
                            threshold,
                        ),
                    )

            for std_fraction in config[
                "gaussian_noise_std_fractions"
            ]:
                for repeat in range(int(config["random_repeats"])):
                    seed = stable_random_seed(
                        base_seed,
                        feature_set,
                        split,
                        "noise",
                        std_fraction,
                        repeat,
                    )
                    modified = apply_gaussian_noise(
                        evaluation,
                        feature_columns,
                        std_fraction=float(std_fraction),
                        standard_deviations=statistics["std"],
                        observed_minimums=statistics["minimum"],
                        observed_maximums=statistics["maximum"],
                        rng=np.random.default_rng(seed),
                    )
                    record(
                        feature_set=feature_set,
                        model_name=model_name,
                        split=split,
                        scenario_group="gaussian_noise",
                        scenario="gaussian_noise",
                        severity=float(std_fraction),
                        sensor="all",
                        repeat=repeat + 1,
                        affected_values=len(evaluation)
                        * len(feature_columns),
                        metrics=_evaluate(
                            estimator,
                            modified,
                            feature_columns,
                            threshold,
                        ),
                    )

            for sensor in feature_columns:
                modified = apply_complete_loss(
                    evaluation,
                    sensor,
                    median=float(statistics["median"][sensor]),
                )
                record(
                    feature_set=feature_set,
                    model_name=model_name,
                    split=split,
                    scenario_group="complete_loss",
                    scenario="median_fallback",
                    severity=1.0,
                    sensor=sensor,
                    repeat=0,
                    affected_values=len(evaluation),
                    metrics=_evaluate(
                        estimator,
                        modified,
                        feature_columns,
                        threshold,
                    ),
                )

                for label, values in statistics["stuck"].items():
                    modified = apply_stuck_sensor(
                        evaluation,
                        sensor,
                        stuck_value=float(values[sensor]),
                    )
                    record(
                        feature_set=feature_set,
                        model_name=model_name,
                        split=split,
                        scenario_group="stuck_sensor",
                        scenario=f"stuck_{label}",
                        severity=label,
                        sensor=sensor,
                        repeat=0,
                        affected_values=len(evaluation),
                        metrics=_evaluate(
                            estimator,
                            modified,
                            feature_columns,
                            threshold,
                        ),
                    )

                for drift_fraction in config["drift_std_fractions"]:
                    modified = apply_gradual_drift(
                        evaluation,
                        sensor,
                        final_std_fraction=float(drift_fraction),
                        training_standard_deviation=float(
                            statistics["std"][sensor]
                        ),
                    )
                    record(
                        feature_set=feature_set,
                        model_name=model_name,
                        split=split,
                        scenario_group="gradual_drift",
                        scenario="gradual_drift",
                        severity=float(drift_fraction),
                        sensor=sensor,
                        repeat=0,
                        affected_values=len(evaluation),
                        metrics=_evaluate(
                            estimator,
                            modified,
                            feature_columns,
                            threshold,
                        ),
                    )

    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[
        metrics["scenario_group"] == "baseline",
        [
            "feature_set",
            "split",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
        ],
    ].rename(
        columns={
            "f1": "baseline_f1",
            "precision": "baseline_precision",
            "recall": "baseline_recall",
            "balanced_accuracy": "baseline_balanced_accuracy",
        }
    )
    metrics = metrics.merge(
        baseline,
        on=["feature_set", "split"],
        how="left",
        validate="many_to_one",
    )
    for metric in ("f1", "precision", "recall", "balanced_accuracy"):
        metrics[f"{metric}_delta"] = (
            metrics[metric] - metrics[f"baseline_{metric}"]
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "robustness_metrics.csv", metrics)

    metadata = {
        "config_path": str(config_source),
        "config_sha256": hashlib.sha256(
            config_source.read_bytes()
        ).hexdigest(),
        "selected_models_path": str(selected_path),
        "selected_models_sha256": hashlib.sha256(
            selected_path.read_bytes()
        ).hexdigest(),
        "comparison_feature_sets": config["comparison_feature_sets"],
        "heldout_splits": config["heldout_splits"],
        "random_seed": base_seed,
        "random_repeats": int(config["random_repeats"]),
        "training_statistics": statistics,
        "scenario_rows": int(len(metrics)),
        "sensor_recommendation_made": False,
    }
    write_json(output / "metadata.json", metadata)
    return {"metrics": metrics, "metadata": metadata}


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured sensor-fault evaluations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--checksums",
        type=Path,
        default=DEFAULT_CHECKSUM_PATH,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--sensor-budget-dir",
        type=Path,
        default=DEFAULT_SENSOR_BUDGET_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    result = run_robustness_evaluation(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        config_path=args.config,
        sensor_budget_dir=args.sensor_budget_dir,
        output_dir=args.output_dir,
    )
    metrics = result["metrics"]
    summary = (
        metrics.groupby(
            ["feature_set", "split", "scenario_group"],
            as_index=False,
        )["f1_delta"]
        .mean()
        .sort_values(["scenario_group", "f1_delta"])
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
