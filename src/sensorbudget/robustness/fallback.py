"""Evaluate a Light-independent fallback under known simulated Light faults."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.provenance import validate_source_checksums
from sensorbudget.data.schema import DEFAULT_CHECKSUM_PATH, DEFAULT_RAW_DIR
from sensorbudget.data.validate import validate_source_splits
from sensorbudget.modeling.artifacts import write_json, write_table
from sensorbudget.modeling.evaluate import evaluate_fitted_model
from sensorbudget.modeling.schema import TARGET_COLUMN
from sensorbudget.modeling.sensor_budget import (
    build_sensor_scenarios,
    load_sensor_budget_config,
)
from sensorbudget.robustness.evaluate import (
    apply_complete_loss,
    apply_gradual_drift,
    apply_light_policy_failure,
    apply_stuck_sensor,
)

DEFAULT_CONFIG_PATH = Path("configs/fallback_mitigation.json")
DEFAULT_SENSOR_BUDGET_CONFIG = Path("configs/sensor_budget.json")
DEFAULT_SENSOR_BUDGET_DIR = Path("models/sensor_budget")
DEFAULT_OUTPUT_DIR = Path("models/fallback_mitigation")


def load_fallback_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the fallback experiment configuration."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not config.get("primary_feature_set"):
        raise ValueError("Fallback config requires primary_feature_set.")
    if not config.get("fallback_candidates"):
        raise ValueError("Fallback config requires fallback_candidates.")
    if not config.get("heldout_splits"):
        raise ValueError("Fallback config requires heldout_splits.")
    quantiles = config.get("stuck_quantiles", {})
    if not quantiles or any(
        not 0 <= float(value) <= 1 for value in quantiles.values()
    ):
        raise ValueError("stuck_quantiles must be between zero and one.")
    if not config.get("drift_std_fractions"):
        raise ValueError("drift_std_fractions cannot be empty.")
    return config


def select_fallback_candidate(
    selected_models: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """Select the candidate with the highest training-only CV mean F1."""

    available = selected_models.loc[
        selected_models["feature_set"].isin(candidates)
    ].copy()
    missing = sorted(set(candidates) - set(available["feature_set"]))
    if missing:
        raise ValueError(f"Unknown fallback candidates: {missing}.")
    ranked = available.sort_values(
        ["cv_f1_mean", "feature_set"],
        ascending=[False, True],
    )
    return ranked.iloc[0]


def validate_fallback_features(
    primary_features: set[str],
    fallback_features: set[str],
) -> None:
    """Require a Light-independent fallback using existing primary sensors."""

    if "Light" in fallback_features:
        raise ValueError("Fallback model must not use Light.")
    if not fallback_features.issubset(primary_features - {"Light"}):
        raise ValueError(
            "Fallback features must already be available in the primary system."
        )


def _load_bundle(
    directory: Path,
    feature_set: str,
    model_name: str,
) -> dict[str, Any]:
    path = directory / f"{feature_set}__{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing fitted model bundle: {path}.")
    return joblib.load(path)


def _metrics(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, float | int]:
    metrics, _ = evaluate_fitted_model(
        bundle["estimator"],
        frame,
        list(bundle["feature_columns"]),
        threshold=float(bundle["threshold"]),
    )
    return metrics


def run_fallback_mitigation(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    sensor_budget_config_path: Path | str = DEFAULT_SENSOR_BUDGET_CONFIG,
    sensor_budget_dir: Path | str = DEFAULT_SENSOR_BUDGET_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Select a fallback with training CV and test oracle-gated mitigation."""

    validate_source_checksums(raw_dir, checksum_path)
    frames = load_source_splits(raw_dir)
    validate_source_splits(frames)

    config_source = Path(config_path)
    config = load_fallback_config(config_source)
    budget_dir = Path(sensor_budget_dir)
    selected_path = budget_dir / "selected_models.csv"
    selected = pd.read_csv(selected_path)

    primary_name = str(config["primary_feature_set"])
    primary_rows = selected.loc[selected["feature_set"] == primary_name]
    if len(primary_rows) != 1:
        raise ValueError(f"Primary feature set {primary_name!r} was not selected.")
    primary_row = primary_rows.iloc[0]
    fallback_row = select_fallback_candidate(
        selected,
        [str(value) for value in config["fallback_candidates"]],
    )

    budget_config = load_sensor_budget_config(sensor_budget_config_path)
    feature_sets, scenarios = build_sensor_scenarios(budget_config)
    primary_features = set(feature_sets[primary_name])
    fallback_features = set(feature_sets[str(fallback_row["feature_set"])])
    validate_fallback_features(primary_features, fallback_features)

    primary_bundle = _load_bundle(
        budget_dir,
        primary_name,
        str(primary_row["selected_model"]),
    )
    fallback_name = str(fallback_row["feature_set"])
    fallback_bundle = _load_bundle(
        budget_dir,
        fallback_name,
        str(fallback_row["selected_model"]),
    )

    training = frames["train"].sort_values("date").reset_index(drop=True)
    light_median = float(training["Light"].median())
    light_std = float(training["Light"].std())
    occupied_light = float(
        training.loc[training[TARGET_COLUMN] == 1, "Light"].median()
    )
    unoccupied_light = float(
        training.loc[training[TARGET_COLUMN] == 0, "Light"].median()
    )
    stuck_values = {
        label: float(training["Light"].quantile(float(quantile)))
        for label, quantile in config["stuck_quantiles"].items()
    }

    rows: list[dict[str, Any]] = []

    def record(
        *,
        split: str,
        scenario_group: str,
        scenario: str,
        severity: float | str,
        strategy: str,
        routed_to_fallback: bool,
        bundle: dict[str, Any],
        evaluation: pd.DataFrame,
    ) -> None:
        rows.append(
            {
                "primary_feature_set": primary_name,
                "fallback_feature_set": fallback_name,
                "split": split,
                "scenario_group": scenario_group,
                "scenario": scenario,
                "severity": severity,
                "strategy": strategy,
                "routed_to_fallback": routed_to_fallback,
                "evaluated_model": str(bundle["model_name"]),
                "evaluated_feature_set": str(bundle["feature_set"]),
                **_metrics(bundle, evaluation),
            }
        )

    for split in config["heldout_splits"]:
        clean = frames[split].sort_values("date").reset_index(drop=True)
        record(
            split=split,
            scenario_group="baseline",
            scenario="clean_primary",
            severity=0.0,
            strategy="primary_clean",
            routed_to_fallback=False,
            bundle=primary_bundle,
            evaluation=clean,
        )
        record(
            split=split,
            scenario_group="baseline",
            scenario="clean_fallback",
            severity=0.0,
            strategy="fallback_only",
            routed_to_fallback=True,
            bundle=fallback_bundle,
            evaluation=clean,
        )

        faults: list[tuple[str, str, float | str, pd.DataFrame]] = []
        faults.append(
            (
                "complete_loss",
                "median_imputation",
                1.0,
                apply_complete_loss(clean, "Light", median=light_median),
            )
        )
        for label, stuck_value in stuck_values.items():
            faults.append(
                (
                    "stuck_sensor",
                    f"stuck_{label}",
                    label,
                    apply_stuck_sensor(clean, "Light", stuck_value=stuck_value),
                )
            )
        for mode in ("unoccupied_lit", "occupied_dark"):
            modified, _ = apply_light_policy_failure(
                clean,
                mode=mode,
                occupied_light_reference=occupied_light,
                unoccupied_light_reference=unoccupied_light,
            )
            faults.append(("light_policy", mode, 1.0, modified))
        for fraction in config["drift_std_fractions"]:
            faults.append(
                (
                    "gradual_drift",
                    "gradual_drift",
                    float(fraction),
                    apply_gradual_drift(
                        clean,
                        "Light",
                        final_std_fraction=float(fraction),
                        training_standard_deviation=light_std,
                    ),
                )
            )

        for scenario_group, scenario, severity, modified in faults:
            record(
                split=split,
                scenario_group=scenario_group,
                scenario=scenario,
                severity=severity,
                strategy="primary_under_fault",
                routed_to_fallback=False,
                bundle=primary_bundle,
                evaluation=modified,
            )
            # Oracle routing knows that the simulated Light fault is active.
            # The fallback ignores Light, so it evaluates the unmodified
            # non-Light columns from the same held-out observations.
            record(
                split=split,
                scenario_group=scenario_group,
                scenario=scenario,
                severity=severity,
                strategy="oracle_gated_fallback",
                routed_to_fallback=True,
                bundle=fallback_bundle,
                evaluation=clean,
            )

    metrics = pd.DataFrame(rows)
    clean_primary = metrics.loc[
        metrics["strategy"] == "primary_clean", ["split", "f1"]
    ].rename(columns={"f1": "clean_primary_f1"})
    metrics = metrics.merge(clean_primary, on="split", validate="many_to_one")
    metrics["f1_delta_from_clean_primary"] = (
        metrics["f1"] - metrics["clean_primary_f1"]
    )

    fault_primary = metrics.loc[
        metrics["strategy"] == "primary_under_fault",
        ["split", "scenario_group", "scenario", "severity", "f1"],
    ].rename(columns={"f1": "faulted_primary_f1"})
    metrics = metrics.merge(
        fault_primary,
        on=["split", "scenario_group", "scenario", "severity"],
        how="left",
        validate="many_to_one",
    )
    metrics["mitigation_f1_gain"] = metrics["f1"] - metrics["faulted_primary_f1"]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "fallback_metrics.csv", metrics)
    candidate_table = selected.loc[
        selected["feature_set"].isin(config["fallback_candidates"]),
        [
            "feature_set",
            "physical_sensors",
            "model_features",
            "selected_model",
            "cv_f1_mean",
        ],
    ].sort_values("cv_f1_mean", ascending=False)
    candidate_table["selected_fallback"] = (
        candidate_table["feature_set"] == fallback_name
    )
    write_table(output / "fallback_candidates.csv", candidate_table)

    primary_scenario = scenarios.set_index("feature_set").loc[primary_name]
    fallback_scenario = scenarios.set_index("feature_set").loc[fallback_name]
    metadata = {
        "config_path": str(config_source),
        "config_sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
        "selection_source": str(selected_path),
        "selection_rule": "highest training-only chronological CV mean F1",
        "primary_feature_set": primary_name,
        "primary_model": str(primary_row["selected_model"]),
        "primary_relative_cost": float(primary_scenario["relative_cost"]),
        "fallback_feature_set": fallback_name,
        "fallback_model": str(fallback_row["selected_model"]),
        "fallback_cv_f1_mean": float(fallback_row["cv_f1_mean"]),
        "fallback_relative_cost": float(fallback_scenario["relative_cost"]),
        "routing_assumption": config["routing_assumption"],
        "hardware_constraint": config["hardware_constraint"],
        "heldout_splits": config["heldout_splits"],
        "sensor_recommendation_made": False,
    }
    write_json(output / "metadata.json", metadata)
    return {
        "candidates": candidate_table,
        "metrics": metrics,
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fallback mitigation experiment from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUM_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--sensor-budget-config",
        type=Path,
        default=DEFAULT_SENSOR_BUDGET_CONFIG,
    )
    parser.add_argument(
        "--sensor-budget-dir", type=Path, default=DEFAULT_SENSOR_BUDGET_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    result = run_fallback_mitigation(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        config_path=args.config,
        sensor_budget_config_path=args.sensor_budget_config,
        sensor_budget_dir=args.sensor_budget_dir,
        output_dir=args.output_dir,
    )
    print("Fallback candidates (selected using training-only CV):")
    print(result["candidates"].to_string(index=False))
    print("\nHeld-out mitigation results:")
    columns = [
        "split",
        "scenario_group",
        "scenario",
        "severity",
        "strategy",
        "f1",
        "f1_delta_from_clean_primary",
        "mitigation_f1_gain",
    ]
    print(result["metrics"][columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
