"""Tune causal Light-health rules and evaluate realistic fallback routing."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.provenance import validate_source_checksums
from sensorbudget.data.schema import DEFAULT_CHECKSUM_PATH, DEFAULT_RAW_DIR
from sensorbudget.data.validate import validate_source_splits
from sensorbudget.modeling.artifacts import write_json, write_table
from sensorbudget.modeling.evaluate import (
    classification_metrics,
    positive_class_probability,
)
from sensorbudget.modeling.schema import TARGET_COLUMN
from sensorbudget.robustness.evaluate import stable_random_seed

DEFAULT_CONFIG_PATH = Path("configs/fault_detection.json")
DEFAULT_SENSOR_BUDGET_DIR = Path("models/sensor_budget")
DEFAULT_OUTPUT_DIR = Path("models/fault_detection")


@dataclass(frozen=True)
class DetectorParameters:
    """Tunable causal Light-health rule parameters."""

    stuck_window: int
    stuck_tolerance_lux: float
    range_low_quantile: float
    range_high_quantile: float
    abrupt_change_quantile: float

    @property
    def candidate_id(self) -> str:
        return (
            f"w{self.stuck_window}_tol{self.stuck_tolerance_lux:g}_"
            f"range{self.range_low_quantile:g}-{self.range_high_quantile:g}_"
            f"jump{self.abrupt_change_quantile:g}"
        )


@dataclass(frozen=True)
class LightReference:
    """Training-derived values used by injection and detection."""

    median: float
    standard_deviation: float
    minimum: float
    maximum: float
    stuck_low: float
    stuck_high: float
    range_low: float
    range_high: float
    abrupt_change_threshold: float


def load_fault_detection_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the fault-detection experiment definition."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required_lists = (
        "fault_scenarios",
        "heldout_splits",
        "selection_episode_lengths",
        "heldout_episode_lengths",
        "stuck_windows",
        "stuck_tolerances_lux",
        "range_quantile_pairs",
        "abrupt_change_quantiles",
    )
    for key in required_lists:
        if not config.get(key):
            raise ValueError(f"{key} cannot be empty.")
    if int(config.get("cv_splits", 0)) < 2:
        raise ValueError("cv_splits must be at least two.")
    if int(config.get("selection_repeats", 0)) < 1:
        raise ValueError("selection_repeats must be at least one.")
    if int(config.get("heldout_repeats", 0)) < 1:
        raise ValueError("heldout_repeats must be at least one.")
    if not 0 <= float(config.get("max_cv_false_positive_rate", -1)) <= 1:
        raise ValueError("max_cv_false_positive_rate must be between zero and one.")
    if any(int(value) < 2 for value in config["stuck_windows"]):
        raise ValueError("stuck_windows must contain values of at least two.")
    if any(float(value) < 0 for value in config["stuck_tolerances_lux"]):
        raise ValueError("stuck_tolerances_lux cannot be negative.")
    for pair in config["range_quantile_pairs"]:
        low, high = float(pair["low"]), float(pair["high"])
        if not 0 <= low < high <= 1:
            raise ValueError("Range quantiles must satisfy 0 <= low < high <= 1.")
    return config


def build_parameter_grid(config: Mapping[str, Any]) -> list[DetectorParameters]:
    """Expand the transparent detector grid from configuration values."""

    parameters = []
    for window, tolerance, pair, abrupt in itertools.product(
        config["stuck_windows"],
        config["stuck_tolerances_lux"],
        config["range_quantile_pairs"],
        config["abrupt_change_quantiles"],
    ):
        parameters.append(
            DetectorParameters(
                stuck_window=int(window),
                stuck_tolerance_lux=float(tolerance),
                range_low_quantile=float(pair["low"]),
                range_high_quantile=float(pair["high"]),
                abrupt_change_quantile=float(abrupt),
            )
        )
    return parameters


def derive_light_reference(
    frame: pd.DataFrame,
    parameters: DetectorParameters,
    stuck_quantiles: Mapping[str, float],
) -> LightReference:
    """Calculate detector thresholds from the supplied training period only."""

    light = frame["Light"].astype(float)
    changes = light.diff().abs().dropna()
    return LightReference(
        median=float(light.median()),
        standard_deviation=float(light.std()),
        minimum=float(light.min()),
        maximum=float(light.max()),
        stuck_low=float(light.quantile(float(stuck_quantiles["low"]))),
        stuck_high=float(light.quantile(float(stuck_quantiles["high"]))),
        range_low=float(light.quantile(parameters.range_low_quantile)),
        range_high=float(light.quantile(parameters.range_high_quantile)),
        abrupt_change_threshold=float(
            changes.quantile(parameters.abrupt_change_quantile)
        ),
    )


def choose_episode_start(
    row_count: int,
    episode_length: int,
    *,
    margin: int,
    rng: np.random.Generator,
) -> int:
    """Choose a reproducible episode start with clean context on both sides."""

    lower = int(margin)
    upper_exclusive = row_count - int(episode_length) - int(margin) + 1
    if upper_exclusive <= lower:
        raise ValueError("Frame is too short for the requested fault episode.")
    return int(rng.integers(lower, upper_exclusive))


def inject_light_fault(
    frame: pd.DataFrame,
    *,
    scenario: str,
    start: int,
    length: int,
    reference: LightReference,
    linear_bias_final_std: float = 1.0,
    out_of_range_std_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inject one Light-only episode and return its row-level ground truth."""

    if start < 0 or length < 1 or start + length > len(frame):
        raise ValueError("Fault episode falls outside the supplied frame.")
    modified = frame.copy()
    truth = pd.Series(False, index=modified.index, name="true_light_fault")
    positions = np.arange(start, start + length)
    truth.iloc[positions] = True

    if scenario == "missing":
        modified.iloc[positions, modified.columns.get_loc("Light")] = np.nan
    elif scenario == "stuck_current":
        stuck_value = float(modified["Light"].iloc[start])
        modified.iloc[positions, modified.columns.get_loc("Light")] = stuck_value
    elif scenario == "stuck_low":
        modified.iloc[positions, modified.columns.get_loc("Light")] = (
            reference.stuck_low
        )
    elif scenario == "stuck_high":
        modified.iloc[positions, modified.columns.get_loc("Light")] = (
            reference.stuck_high
        )
    elif scenario == "out_of_range_high":
        value = reference.maximum + (
            out_of_range_std_multiplier * reference.standard_deviation
        )
        modified.iloc[positions, modified.columns.get_loc("Light")] = value
    elif scenario in {"linear_bias_positive", "linear_bias_negative"}:
        direction = 1.0 if scenario.endswith("positive") else -1.0
        offsets = np.linspace(
            0.0,
            direction
            * linear_bias_final_std
            * reference.standard_deviation,
            num=length,
        )
        modified.iloc[positions, modified.columns.get_loc("Light")] = (
            modified["Light"].iloc[positions].to_numpy() + offsets
        )
    else:
        raise ValueError(f"Unknown Light fault scenario: {scenario!r}.")
    return modified, truth


def detect_light_faults(
    light: pd.Series,
    *,
    parameters: DetectorParameters,
    reference: LightReference,
) -> pd.DataFrame:
    """Apply causal rules using only each current and preceding Light value."""

    values = light.astype(float)
    missing = values.isna()
    rolling = values.rolling(
        window=parameters.stuck_window,
        min_periods=parameters.stuck_window,
    )
    rolling_range = rolling.max() - rolling.min()
    # Darkness is frequently a legitimate, constant zero in this dataset.
    # Treating it as a stuck sensor would route most unoccupied nights to the
    # fallback. A constant signal is therefore suspicious only above the
    # normal low-light reference. This deliberately makes stuck-at-dark faults
    # difficult to detect without another source of context.
    rolling_mean = rolling.mean()
    stuck = (
        rolling_range.le(parameters.stuck_tolerance_lux)
        & rolling_mean.gt(reference.stuck_low + parameters.stuck_tolerance_lux)
        & ~missing
    )
    outside_range = (
        values.lt(reference.range_low) | values.gt(reference.range_high)
    ) & ~missing
    abrupt_change = values.diff().abs().gt(reference.abrupt_change_threshold)
    detected = missing | stuck | outside_range | abrupt_change
    return pd.DataFrame(
        {
            "missing_rule": missing,
            "stuck_rule": stuck,
            "range_rule": outside_range,
            "abrupt_rule": abrupt_change,
            "detected_light_fault": detected,
            "rolling_light_range": rolling_range,
            "rolling_light_mean": rolling_mean,
        },
        index=light.index,
    )


def fault_detection_metrics(
    truth: pd.Series | np.ndarray,
    detected: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Calculate fault classification, false alarm, and delay metrics."""

    actual = np.asarray(truth, dtype=bool)
    predicted = np.asarray(detected, dtype=bool)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="binary",
        zero_division=0,
    )
    negative = ~actual
    false_positive_rate = (
        float(predicted[negative].mean()) if negative.any() else 0.0
    )
    true_positions = np.flatnonzero(actual)
    delay = 0
    if true_positions.size:
        episode_predictions = np.flatnonzero(predicted[true_positions])
        delay = (
            int(episode_predictions[0])
            if episode_predictions.size
            else int(true_positions.size)
        )
    return {
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_f1": float(f1),
        "false_positive_rate": false_positive_rate,
        "detection_delay_rows": delay,
        "true_fault_rows": int(actual.sum()),
        "detected_rows": int(predicted.sum()),
    }


def route_probabilities(
    primary_probability: np.ndarray,
    fallback_probability: np.ndarray,
    route_to_fallback: pd.Series | np.ndarray,
) -> np.ndarray:
    """Choose fallback probabilities exactly where the routing mask is true."""

    primary = np.asarray(primary_probability, dtype=float)
    fallback = np.asarray(fallback_probability, dtype=float)
    route = np.asarray(route_to_fallback, dtype=bool)
    if not (len(primary) == len(fallback) == len(route)):
        raise ValueError("Probability arrays and routing mask must have equal length.")
    return np.where(route, fallback, primary)


def _load_bundle(
    budget_dir: Path,
    selected: pd.DataFrame,
    feature_set: str,
) -> dict[str, Any]:
    rows = selected.loc[selected["feature_set"] == feature_set]
    if len(rows) != 1:
        raise ValueError(f"Unknown selected feature set: {feature_set!r}.")
    model_name = str(rows.iloc[0]["selected_model"])
    path = budget_dir / f"{feature_set}__{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing fitted model bundle: {path}.")
    return joblib.load(path)


def _predict_probabilities(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    light_median: float | None = None,
) -> np.ndarray:
    features = list(bundle["feature_columns"])
    model_input = frame[features].copy()
    if light_median is not None and "Light" in model_input:
        model_input["Light"] = model_input["Light"].fillna(light_median)
    return positive_class_probability(bundle["estimator"], model_input)


def _parameter_record(parameters: DetectorParameters) -> dict[str, Any]:
    return {
        "candidate_id": parameters.candidate_id,
        "stuck_window": parameters.stuck_window,
        "stuck_tolerance_lux": parameters.stuck_tolerance_lux,
        "range_low_quantile": parameters.range_low_quantile,
        "range_high_quantile": parameters.range_high_quantile,
        "abrupt_change_quantile": parameters.abrupt_change_quantile,
    }


def tune_detector(
    training: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, DetectorParameters]:
    """Select detector parameters on synthetic chronological validation faults."""

    ordered = training.sort_values("date").reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=int(config["cv_splits"]))
    grid = build_parameter_grid(config)
    rows: list[dict[str, Any]] = []
    max_window = max(parameter.stuck_window for parameter in grid)
    base_seed = int(config["random_seed"])

    for fold, (development_indices, validation_indices) in enumerate(
        splitter.split(ordered), start=1
    ):
        development = ordered.iloc[development_indices]
        validation = ordered.iloc[validation_indices].reset_index(drop=True)
        for parameters in grid:
            reference = derive_light_reference(
                development,
                parameters,
                config["stuck_reference_quantiles"],
            )
            for scenario in config["fault_scenarios"]:
                for length in config["selection_episode_lengths"]:
                    for repeat in range(int(config["selection_repeats"])):
                        seed = stable_random_seed(
                            base_seed,
                            "cv",
                            fold,
                            scenario,
                            length,
                            repeat,
                        )
                        start = choose_episode_start(
                            len(validation),
                            int(length),
                            margin=max_window + 2,
                            rng=np.random.default_rng(seed),
                        )
                        modified, truth = inject_light_fault(
                            validation,
                            scenario=str(scenario),
                            start=start,
                            length=int(length),
                            reference=reference,
                            linear_bias_final_std=float(
                                config["linear_bias_final_std"]
                            ),
                            out_of_range_std_multiplier=float(
                                config["out_of_range_std_multiplier"]
                            ),
                        )
                        detection = detect_light_faults(
                            modified["Light"],
                            parameters=parameters,
                            reference=reference,
                        )
                        rows.append(
                            {
                                **_parameter_record(parameters),
                                "fold": fold,
                                "scenario": scenario,
                                "episode_length": int(length),
                                "repeat": repeat + 1,
                                **fault_detection_metrics(
                                    truth,
                                    detection["detected_light_fault"],
                                ),
                            }
                        )

    details = pd.DataFrame(rows)
    group_columns = [
        "candidate_id",
        "stuck_window",
        "stuck_tolerance_lux",
        "range_low_quantile",
        "range_high_quantile",
        "abrupt_change_quantile",
    ]
    metric_columns = [
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "false_positive_rate",
        "detection_delay_rows",
    ]
    summary = details.groupby(group_columns, as_index=False)[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(column).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary["eligible"] = (
        summary["false_positive_rate_mean"]
        <= float(config["max_cv_false_positive_rate"])
    )
    selection_pool = summary.loc[summary["eligible"]].copy()
    if selection_pool.empty:
        selection_pool = summary.copy()
    ranked_pool = selection_pool.sort_values(
        [
            "detection_f1_mean",
            "false_positive_rate_mean",
            "detection_delay_rows_mean",
            "candidate_id",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    best = ranked_pool.iloc[0]
    selected = DetectorParameters(
        stuck_window=int(best["stuck_window"]),
        stuck_tolerance_lux=float(best["stuck_tolerance_lux"]),
        range_low_quantile=float(best["range_low_quantile"]),
        range_high_quantile=float(best["range_high_quantile"]),
        abrupt_change_quantile=float(best["abrupt_change_quantile"]),
    )
    summary["selected"] = summary["candidate_id"] == selected.candidate_id
    summary = summary.sort_values(
        [
            "eligible",
            "detection_f1_mean",
            "false_positive_rate_mean",
            "detection_delay_rows_mean",
            "candidate_id",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    return summary, selected


def run_fault_detection_evaluation(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    sensor_budget_dir: Path | str = DEFAULT_SENSOR_BUDGET_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Tune causal health rules and evaluate non-oracle held-out routing."""

    validate_source_checksums(raw_dir, checksum_path)
    frames = load_source_splits(raw_dir)
    validate_source_splits(frames)
    config_source = Path(config_path)
    config = load_fault_detection_config(config_source)
    training = frames["train"].sort_values("date").reset_index(drop=True)

    cv_results, selected_parameters = tune_detector(training, config)
    final_reference = derive_light_reference(
        training,
        selected_parameters,
        config["stuck_reference_quantiles"],
    )

    budget_dir = Path(sensor_budget_dir)
    selected_path = budget_dir / "selected_models.csv"
    selected_models = pd.read_csv(selected_path)
    primary_bundle = _load_bundle(
        budget_dir,
        selected_models,
        str(config["primary_feature_set"]),
    )
    fallback_bundle = _load_bundle(
        budget_dir,
        selected_models,
        str(config["fallback_feature_set"]),
    )
    if "Light" in fallback_bundle["feature_columns"]:
        raise ValueError("Configured fallback model must not use Light.")
    routing_threshold = float(primary_bundle["threshold"])
    if float(fallback_bundle["threshold"]) != routing_threshold:
        raise ValueError(
            "Primary and fallback bundles must use the same routing threshold."
        )

    detection_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    trace_frames: list[pd.DataFrame] = []
    base_seed = int(config["random_seed"])
    margin = selected_parameters.stuck_window + 2

    for split in config["heldout_splits"]:
        clean = frames[split].sort_values("date").reset_index(drop=True)
        fallback_probability = _predict_probabilities(fallback_bundle, clean)
        clean_primary_probability = _predict_probabilities(
            primary_bundle,
            clean,
            light_median=final_reference.median,
        )
        clean_detection = detect_light_faults(
            clean["Light"],
            parameters=selected_parameters,
            reference=final_reference,
        )
        clean_truth = pd.Series(False, index=clean.index)
        detection_rows.append(
            {
                "split": split,
                "scenario": "clean",
                "episode_length": 0,
                "repeat": 0,
                "episode_start_row": -1,
                "episode_end_row": -1,
                **fault_detection_metrics(
                    clean_truth,
                    clean_detection["detected_light_fault"],
                ),
            }
        )
        clean_route_masks = {
            "primary_only": np.zeros(len(clean), dtype=bool),
            "fallback_only": np.ones(len(clean), dtype=bool),
            "oracle_routing": np.zeros(len(clean), dtype=bool),
            "detector_routing": clean_detection[
                "detected_light_fault"
            ].to_numpy(),
        }
        for strategy, route_mask in clean_route_masks.items():
            probability = route_probabilities(
                clean_primary_probability,
                fallback_probability,
                route_mask,
            )
            overall = classification_metrics(
                clean[TARGET_COLUMN],
                probability,
                threshold=routing_threshold,
            )
            routing_rows.append(
                {
                    "split": split,
                    "scenario": "clean",
                    "episode_length": 0,
                    "repeat": 0,
                    "strategy": strategy,
                    "routed_rows": int(np.asarray(route_mask).sum()),
                    "routed_fraction": float(np.asarray(route_mask).mean()),
                    **overall,
                    **{
                        f"fault_window_{key}": float("nan")
                        for key in overall
                    },
                }
            )
        for scenario in config["fault_scenarios"]:
            for length in config["heldout_episode_lengths"]:
                for repeat in range(int(config["heldout_repeats"])):
                    seed = stable_random_seed(
                        base_seed,
                        "heldout",
                        split,
                        scenario,
                        length,
                        repeat,
                    )
                    start = choose_episode_start(
                        len(clean),
                        int(length),
                        margin=margin,
                        rng=np.random.default_rng(seed),
                    )
                    modified, truth = inject_light_fault(
                        clean,
                        scenario=str(scenario),
                        start=start,
                        length=int(length),
                        reference=final_reference,
                        linear_bias_final_std=float(
                            config["linear_bias_final_std"]
                        ),
                        out_of_range_std_multiplier=float(
                            config["out_of_range_std_multiplier"]
                        ),
                    )
                    detection = detect_light_faults(
                        modified["Light"],
                        parameters=selected_parameters,
                        reference=final_reference,
                    )
                    detected = detection["detected_light_fault"]
                    detection_metric = fault_detection_metrics(truth, detected)
                    detection_rows.append(
                        {
                            "split": split,
                            "scenario": scenario,
                            "episode_length": int(length),
                            "repeat": repeat + 1,
                            "episode_start_row": start,
                            "episode_end_row": start + int(length) - 1,
                            **detection_metric,
                        }
                    )

                    primary_probability = _predict_probabilities(
                        primary_bundle,
                        modified,
                        light_median=final_reference.median,
                    )
                    route_masks = {
                        "primary_only": np.zeros(len(clean), dtype=bool),
                        "fallback_only": np.ones(len(clean), dtype=bool),
                        "oracle_routing": truth.to_numpy(),
                        "detector_routing": detected.to_numpy(),
                    }
                    for strategy, route_mask in route_masks.items():
                        probability = route_probabilities(
                            primary_probability,
                            fallback_probability,
                            route_mask,
                        )
                        overall = classification_metrics(
                            clean[TARGET_COLUMN],
                            probability,
                            threshold=routing_threshold,
                        )
                        fault_window = classification_metrics(
                            clean.loc[truth, TARGET_COLUMN],
                            probability[truth.to_numpy()],
                            threshold=routing_threshold,
                        )
                        routing_rows.append(
                            {
                                "split": split,
                                "scenario": scenario,
                                "episode_length": int(length),
                                "repeat": repeat + 1,
                                "strategy": strategy,
                                "routed_rows": int(np.asarray(route_mask).sum()),
                                "routed_fraction": float(
                                    np.asarray(route_mask).mean()
                                ),
                                **overall,
                                **{
                                    f"fault_window_{key}": value
                                    for key, value in fault_window.items()
                                },
                            }
                        )

                    if repeat == 0 and int(length) == int(
                        config["heldout_episode_lengths"][0]
                    ):
                        trace_start = max(0, start - 15)
                        trace_end = min(len(clean), start + int(length) + 15)
                        trace = modified.iloc[trace_start:trace_end][
                            ["date", "Light", TARGET_COLUMN]
                        ].copy()
                        trace.insert(0, "split", split)
                        trace.insert(1, "scenario", scenario)
                        trace["true_light_fault"] = truth.iloc[
                            trace_start:trace_end
                        ].to_numpy()
                        for column in (
                            "missing_rule",
                            "stuck_rule",
                            "range_rule",
                            "abrupt_rule",
                            "detected_light_fault",
                        ):
                            trace[column] = detection[column].iloc[
                                trace_start:trace_end
                            ].to_numpy()
                        trace["primary_probability"] = primary_probability[
                            trace_start:trace_end
                        ]
                        trace["fallback_probability"] = fallback_probability[
                            trace_start:trace_end
                        ]
                        trace_frames.append(trace)

    detection_metrics = pd.DataFrame(detection_rows)
    routing_metrics = pd.DataFrame(routing_rows)
    example_trace = pd.concat(trace_frames, ignore_index=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "detector_cv_results.csv", cv_results)
    write_table(output / "heldout_detection_metrics.csv", detection_metrics)
    write_table(output / "heldout_routing_metrics.csv", routing_metrics)
    write_table(output / "example_episode_trace.csv", example_trace)

    selected_cv = cv_results.loc[cv_results["selected"]].iloc[0]
    metadata = {
        "config_path": str(config_source),
        "config_sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
        "selection_source": (
            "synthetic episodes in chronological training-validation folds"
        ),
        "selection_rule": config["selection_rule"],
        "selected_detector": _parameter_record(selected_parameters),
        "selected_cv_metrics": {
            "detection_precision_mean": float(
                selected_cv["detection_precision_mean"]
            ),
            "detection_recall_mean": float(selected_cv["detection_recall_mean"]),
            "detection_f1_mean": float(selected_cv["detection_f1_mean"]),
            "false_positive_rate_mean": float(
                selected_cv["false_positive_rate_mean"]
            ),
            "detection_delay_rows_mean": float(
                selected_cv["detection_delay_rows_mean"]
            ),
        },
        "training_light_reference": final_reference.__dict__,
        "primary_feature_set": config["primary_feature_set"],
        "primary_model": primary_bundle["model_name"],
        "fallback_feature_set": config["fallback_feature_set"],
        "fallback_model": fallback_bundle["model_name"],
        "scope_note": config["scope_note"],
        "sensor_recommendation_made": False,
    }
    write_json(output / "metadata.json", metadata)
    return {
        "cv_results": cv_results,
        "detection_metrics": detection_metrics,
        "routing_metrics": routing_metrics,
        "example_trace": example_trace,
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run fault-detector selection and held-out routing evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUM_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--sensor-budget-dir", type=Path, default=DEFAULT_SENSOR_BUDGET_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    result = run_fault_detection_evaluation(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        config_path=args.config,
        sensor_budget_dir=args.sensor_budget_dir,
        output_dir=args.output_dir,
    )
    print("Selected detector:")
    print(json.dumps(result["metadata"]["selected_detector"], indent=2))
    print("\nHeld-out detection summary:")
    print(
        result["detection_metrics"]
        .groupby(["split", "scenario"], as_index=False)[
            [
                "detection_precision",
                "detection_recall",
                "detection_f1",
                "false_positive_rate",
                "detection_delay_rows",
            ]
        ]
        .mean()
        .to_string(index=False)
    )
    print("\nHeld-out routing F1:")
    print(
        result["routing_metrics"]
        .groupby(["split", "scenario", "strategy"], as_index=False)["f1"]
        .mean()
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
