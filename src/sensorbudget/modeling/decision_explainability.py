"""Select operating thresholds and explain the selected occupancy model."""

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
from sklearn.metrics import brier_score_loss

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.provenance import validate_source_checksums
from sensorbudget.data.schema import DEFAULT_CHECKSUM_PATH, DEFAULT_RAW_DIR
from sensorbudget.data.validate import validate_source_splits
from sensorbudget.modeling.artifacts import write_json, write_table
from sensorbudget.modeling.evaluate import (
    classification_metrics,
    positive_class_probability,
    time_series_cross_validate,
)
from sensorbudget.modeling.schema import TARGET_COLUMN

DEFAULT_CONFIG_PATH = Path("configs/decision_explainability.json")
DEFAULT_SENSOR_BUDGET_DIR = Path("models/sensor_budget")
DEFAULT_OUTPUT_DIR = Path("models/decision_explainability")


def load_decision_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the Phase 6 decision configuration."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = config.get("cost_scenarios", {})
    if not scenarios:
        raise ValueError("cost_scenarios cannot be empty.")
    if config.get("operating_scenario") not in scenarios:
        raise ValueError("operating_scenario must name a configured scenario.")
    for name, scenario in scenarios.items():
        if float(scenario["false_positive_cost"]) <= 0:
            raise ValueError(f"{name} false_positive_cost must be positive.")
        if float(scenario["false_negative_cost"]) <= 0:
            raise ValueError(f"{name} false_negative_cost must be positive.")
    start = float(config["threshold_start"])
    stop = float(config["threshold_stop"])
    step = float(config["threshold_step"])
    if not 0 < start < stop < 1 or step <= 0:
        raise ValueError("Threshold grid must satisfy 0 < start < stop < 1.")
    if int(config.get("calibration_bins", 0)) < 2:
        raise ValueError("calibration_bins must be at least two.")
    return config


def threshold_grid(config: Mapping[str, Any]) -> np.ndarray:
    """Construct the inclusive configured probability-threshold grid."""

    start = float(config["threshold_start"])
    stop = float(config["threshold_stop"])
    step = float(config["threshold_step"])
    count = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, count), 10)


def evaluate_thresholds(
    target: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    thresholds: Sequence[float],
    cost_scenarios: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Calculate classification metrics and assumed cost for every threshold."""

    rows = []
    row_count = len(target)
    for threshold in thresholds:
        metrics = classification_metrics(
            target, np.asarray(probability), threshold=float(threshold)
        )
        for name, scenario in cost_scenarios.items():
            total_cost = (
                metrics["false_positive"] * float(scenario["false_positive_cost"])
                + metrics["false_negative"]
                * float(scenario["false_negative_cost"])
            )
            rows.append(
                {
                    "scenario": name,
                    "scenario_label": scenario["label"],
                    "false_positive_cost": float(
                        scenario["false_positive_cost"]
                    ),
                    "false_negative_cost": float(
                        scenario["false_negative_cost"]
                    ),
                    "cost_per_1000_rows": float(total_cost / row_count * 1000),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def select_operating_thresholds(curves: pd.DataFrame) -> pd.DataFrame:
    """Select the minimum-cost validation threshold for each cost scenario."""

    selected = []
    for scenario, candidates in curves.groupby("scenario"):
        default_row = candidates.iloc[(candidates["threshold"] - 0.5).abs().argmin()]
        ranked = candidates.assign(
            distance_from_default=(candidates["threshold"] - 0.5).abs()
        ).sort_values(
            ["cost_per_1000_rows", "f1", "distance_from_default"],
            ascending=[True, False, True],
        )
        best = ranked.iloc[0].drop(labels="distance_from_default").copy()
        best["default_threshold"] = float(default_row["threshold"])
        best["default_cost_per_1000_rows"] = float(
            default_row["cost_per_1000_rows"]
        )
        best["validation_cost_reduction_vs_default"] = float(
            default_row["cost_per_1000_rows"] - best["cost_per_1000_rows"]
        )
        selected.append(best)
    return pd.DataFrame(selected).sort_values("scenario").reset_index(drop=True)


def calibration_table(
    target: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    *,
    bins: int,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build an equal-width reliability table and summary statistics."""

    frame = pd.DataFrame(
        {
            "actual": np.asarray(target, dtype=int),
            "probability": np.asarray(probability, dtype=float),
        }
    )
    edges = np.linspace(0, 1, bins + 1)
    frame["bin"] = pd.cut(
        frame["probability"], edges, include_lowest=True, right=True
    )
    grouped = (
        frame.groupby("bin", observed=False)
        .agg(
            row_count=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_occupancy_rate=("actual", "mean"),
        )
        .reset_index()
    )
    grouped["bin_lower"] = grouped["bin"].apply(lambda value: float(value.left))
    grouped["bin_upper"] = grouped["bin"].apply(lambda value: float(value.right))
    grouped.insert(0, "dataset", dataset)
    grouped = grouped.drop(columns="bin")
    populated = grouped.loc[grouped["row_count"] > 0].copy()
    absolute_gap = (
        populated["mean_predicted_probability"]
        - populated["observed_occupancy_rate"]
    ).abs()
    ece = float((absolute_gap * populated["row_count"]).sum() / len(frame))
    summary = {
        "brier_score": float(
            brier_score_loss(frame["actual"], frame["probability"])
        ),
        "expected_calibration_error": ece,
    }
    return grouped, summary


def coefficient_table(bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Extract standardized logistic coefficients and one-SD odds ratios."""

    estimator = bundle["estimator"]
    if not hasattr(estimator, "named_steps"):
        raise TypeError("Global coefficient explanation requires a fitted pipeline.")
    scaler = estimator.named_steps["scale"]
    model = estimator.named_steps["model"]
    features = list(bundle["feature_columns"])
    coefficients = np.asarray(model.coef_[0], dtype=float)
    return pd.DataFrame(
        {
            "feature": features,
            "training_mean": scaler.mean_,
            "training_standard_deviation": scaler.scale_,
            "standardized_coefficient": coefficients,
            "odds_ratio_per_training_sd": np.exp(coefficients),
            "coefficient_per_original_unit": coefficients / scaler.scale_,
        }
    ).sort_values("standardized_coefficient", ascending=False)


def add_prediction_outcomes(
    frame: pd.DataFrame, probability: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Attach probabilities, decisions, and TP/TN/FP/FN labels."""

    result = frame.copy()
    result["probability_occupied"] = probability
    result["predicted_occupancy"] = (probability >= threshold).astype(int)
    actual = result[TARGET_COLUMN].to_numpy()
    predicted = result["predicted_occupancy"].to_numpy()
    result["outcome"] = np.select(
        [
            (actual == 1) & (predicted == 1),
            (actual == 0) & (predicted == 0),
            (actual == 0) & (predicted == 1),
            (actual == 1) & (predicted == 0),
        ],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown",
    )
    return result


def representative_explanations(
    bundle: Mapping[str, Any],
    predictions: pd.DataFrame,
    *,
    threshold: float,
    rows_per_outcome: int,
) -> pd.DataFrame:
    """Explain representative decisions with additive logistic contributions."""

    estimator = bundle["estimator"]
    scaler = estimator.named_steps["scale"]
    model = estimator.named_steps["model"]
    features = list(bundle["feature_columns"])
    candidates = predictions.assign(
        distance_from_threshold=(predictions["probability_occupied"] - threshold).abs()
    )
    representatives = (
        candidates.sort_values("distance_from_threshold")
        .groupby(["source_split", "outcome"], as_index=False, group_keys=False)
        .head(rows_per_outcome)
        .copy()
    )
    standardized = scaler.transform(representatives[features])
    contributions = standardized * np.asarray(model.coef_[0])
    representatives["intercept_contribution"] = float(model.intercept_[0])
    for index, feature in enumerate(features):
        representatives[f"{feature}_contribution"] = contributions[:, index]
    representatives["logit_from_contributions"] = (
        contributions.sum(axis=1) + float(model.intercept_[0])
    )
    columns = [
        "source_split",
        "source_row_id",
        "date",
        TARGET_COLUMN,
        "predicted_occupancy",
        "probability_occupied",
        "outcome",
        "distance_from_threshold",
        *features,
        "intercept_contribution",
        *[f"{feature}_contribution" for feature in features],
        "logit_from_contributions",
    ]
    return representatives[columns].sort_values(
        ["source_split", "outcome", "distance_from_threshold"]
    )


def occupancy_phase_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize occupied-row recall by elapsed time after each occupancy onset."""

    rows = []
    labels = ["0–2 min", "3–5 min", "6–15 min", ">15 min"]
    for split, split_rows in predictions.groupby("source_split"):
        ordered = split_rows.sort_values("date").copy()
        state_group = ordered[TARGET_COLUMN].ne(ordered[TARGET_COLUMN].shift()).cumsum()
        group_start = ordered.groupby(state_group)["date"].transform("min")
        ordered["minutes_since_state_change"] = (
            (ordered["date"] - group_start).dt.total_seconds() / 60
        )
        occupied = ordered.loc[ordered[TARGET_COLUMN] == 1].copy()
        occupied["occupancy_phase"] = pd.cut(
            occupied["minutes_since_state_change"],
            bins=[-np.inf, 2, 5, 15, np.inf],
            labels=labels,
        )
        summary = (
            occupied.groupby("occupancy_phase", observed=False)
            .agg(
                occupied_rows=(TARGET_COLUMN, "size"),
                true_positive=("predicted_occupancy", "sum"),
            )
            .reset_index()
        )
        summary["false_negative"] = (
            summary["occupied_rows"] - summary["true_positive"]
        )
        summary["recall"] = summary["true_positive"].div(
            summary["occupied_rows"].replace(0, np.nan)
        )
        summary.insert(0, "split", split)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_decision_explainability(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    sensor_budget_dir: Path | str = DEFAULT_SENSOR_BUDGET_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run validation-only operating-point selection and explanation analysis."""

    validate_source_checksums(raw_dir, checksum_path)
    frames = load_source_splits(raw_dir)
    validate_source_splits(frames)
    config_source = Path(config_path)
    config = load_decision_config(config_source)
    budget_dir = Path(sensor_budget_dir)
    selected_models = pd.read_csv(budget_dir / "selected_models.csv")
    feature_set = str(config["feature_set"])
    selected_row = selected_models.loc[selected_models["feature_set"] == feature_set]
    if len(selected_row) != 1:
        raise ValueError(f"Unknown selected feature set: {feature_set!r}.")
    model_name = str(selected_row.iloc[0]["selected_model"])
    bundle_path = budget_dir / f"{feature_set}__{model_name}.joblib"
    bundle = joblib.load(bundle_path)
    training = frames["train"].sort_values("date").reset_index(drop=True)

    _, cv_predictions = time_series_cross_validate(
        training,
        {model_name: bundle["estimator"]},
        {feature_set: list(bundle["feature_columns"])},
        n_splits=int(config["cv_splits"]),
        return_predictions=True,
    )
    curves = evaluate_thresholds(
        cv_predictions[TARGET_COLUMN],
        cv_predictions["probability_occupied"],
        threshold_grid(config),
        config["cost_scenarios"],
    )
    selected_thresholds = select_operating_thresholds(curves)
    operating_scenario = str(config["operating_scenario"])
    operating_threshold = float(
        selected_thresholds.loc[
            selected_thresholds["scenario"] == operating_scenario, "threshold"
        ].iloc[0]
    )

    calibration_frames = []
    calibration_summaries = []
    cv_calibration, cv_summary = calibration_table(
        cv_predictions[TARGET_COLUMN],
        cv_predictions["probability_occupied"],
        bins=int(config["calibration_bins"]),
        dataset="chronological_cv",
    )
    calibration_frames.append(cv_calibration)
    calibration_summaries.append({"dataset": "chronological_cv", **cv_summary})
    heldout_metric_rows = []
    all_operating_predictions = []

    for split in config.get("heldout_splits", ["test_1", "test_2"]):
        evaluation = frames[split].sort_values("date").reset_index(drop=True)
        probability = positive_class_probability(
            bundle["estimator"], evaluation[list(bundle["feature_columns"])]
        )
        split_calibration, split_summary = calibration_table(
            evaluation[TARGET_COLUMN],
            probability,
            bins=int(config["calibration_bins"]),
            dataset=split,
        )
        calibration_frames.append(split_calibration)
        calibration_summaries.append({"dataset": split, **split_summary})
        for row in selected_thresholds.itertuples(index=False):
            scenario = config["cost_scenarios"][row.scenario]
            for threshold_source, threshold in (
                ("validation_selected", float(row.threshold)),
                ("default_0.5", 0.5),
            ):
                metrics = classification_metrics(
                    evaluation[TARGET_COLUMN], probability, threshold=threshold
                )
                cost = (
                    metrics["false_positive"]
                    * float(scenario["false_positive_cost"])
                    + metrics["false_negative"]
                    * float(scenario["false_negative_cost"])
                ) / len(evaluation) * 1000
                heldout_metric_rows.append(
                    {
                        "split": split,
                        "scenario": row.scenario,
                        "scenario_label": row.scenario_label,
                        "threshold_source": threshold_source,
                        "cost_per_1000_rows": float(cost),
                        **metrics,
                    }
                )
        operating_predictions = add_prediction_outcomes(
            evaluation, probability, operating_threshold
        )
        all_operating_predictions.append(operating_predictions)

    operating_predictions = pd.concat(all_operating_predictions, ignore_index=True)
    local = representative_explanations(
        bundle,
        operating_predictions,
        threshold=operating_threshold,
        rows_per_outcome=int(config["representative_rows_per_outcome"]),
    )
    phase = occupancy_phase_table(operating_predictions)
    coefficients = coefficient_table(bundle)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "cv_predictions": cv_predictions,
        "threshold_curves": curves,
        "selected_thresholds": selected_thresholds,
        "heldout_metrics": pd.DataFrame(heldout_metric_rows),
        "calibration_bins": pd.concat(calibration_frames, ignore_index=True),
        "calibration_summary": pd.DataFrame(calibration_summaries),
        "global_coefficients": coefficients,
        "representative_explanations": local,
        "transition_metrics": phase,
    }
    for name, frame in artifacts.items():
        write_table(output / f"{name}.csv", frame)
    metadata = {
        "config_path": str(config_source),
        "config_sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
        "feature_set": feature_set,
        "model": model_name,
        "feature_columns": list(bundle["feature_columns"]),
        "selection_source": "out-of-fold chronological training predictions only",
        "selection_rule": config["selection_rule"],
        "operating_scenario": operating_scenario,
        "operating_threshold": operating_threshold,
        "heldout_data_used_for_threshold_selection": False,
        "probability_recalibration_applied": False,
        "calibration_note": config["calibration_note"],
    }
    write_json(output / "metadata.json", metadata)
    return {**artifacts, "metadata": metadata}


def main(argv: Sequence[str] | None = None) -> int:
    """Run Phase 6 decision and explainability analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUM_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_decision_explainability(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print("Validation-selected thresholds:")
    print(
        result["selected_thresholds"][[
            "scenario_label", "threshold", "cost_per_1000_rows", "precision",
            "recall", "f1",
        ]].to_string(index=False)
    )
    print("\nCalibration summary:")
    print(result["calibration_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
