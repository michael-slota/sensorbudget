"""Train and evaluate occupancy models with training-only Light faults."""

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
from sklearn.base import BaseEstimator, clone
from threadpoolctl import threadpool_limits

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.provenance import validate_source_checksums
from sensorbudget.data.schema import DEFAULT_CHECKSUM_PATH, DEFAULT_RAW_DIR
from sensorbudget.data.validate import validate_source_splits
from sensorbudget.modeling.artifacts import write_json, write_table
from sensorbudget.modeling.evaluate import (
    classification_metrics,
    positive_class_probability,
)
from sensorbudget.modeling.models import build_baseline_estimators
from sensorbudget.modeling.schema import DEFAULT_THRESHOLD, TARGET_COLUMN
from sensorbudget.robustness.evaluate import stable_random_seed
from sensorbudget.robustness.fault_detection import (
    DetectorParameters,
    choose_episode_start,
    derive_light_reference,
    detect_light_faults,
    inject_light_fault,
    route_probabilities,
)

DEFAULT_CONFIG_PATH = Path("configs/fault_aware_training.json")
DEFAULT_FAULT_DETECTION_CONFIG_PATH = Path("configs/fault_detection.json")
DEFAULT_SENSOR_BUDGET_DIR = Path("models/sensor_budget")
DEFAULT_FAULT_DETECTION_DIR = Path("models/fault_detection")
DEFAULT_OUTPUT_DIR = Path("models/fault_aware")
PRIMARY_COLUMNS = ["Temperature", "Light", "CO2"]
MISSING_INDICATOR = "Light_missing"


def load_fault_aware_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the fault-aware experiment definition."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "candidate_models",
        "representations",
        "augmentation_ratios",
        "fault_scenarios",
        "heldout_splits",
        "heldout_episode_lengths",
    ):
        if not config.get(key):
            raise ValueError(f"{key} cannot be empty.")
    if int(config.get("cv_splits", 0)) < 2:
        raise ValueError("cv_splits must be at least two.")
    if int(config.get("heldout_repeats", 0)) < 1:
        raise ValueError("heldout_repeats must be at least one.")
    if not 0 <= float(config.get("maximum_clean_f1_drop", -1)) <= 1:
        raise ValueError("maximum_clean_f1_drop must be between zero and one.")
    if any(not 0 < float(value) <= 1 for value in config["augmentation_ratios"]):
        raise ValueError(
            "augmentation_ratios must be greater than zero and at most one."
        )
    allowed = {"fault_aware", "fault_aware_missing_indicator"}
    unknown = set(config["representations"]) - allowed
    if unknown:
        raise ValueError(f"Unknown representations: {sorted(unknown)}.")
    return config


def prepare_model_input(
    frame: pd.DataFrame,
    *,
    light_median: float,
    include_missing_indicator: bool,
) -> pd.DataFrame:
    """Create model inputs without allowing missing values into estimators."""

    prepared = frame[PRIMARY_COLUMNS].copy()
    missing = prepared["Light"].isna()
    prepared["Light"] = prepared["Light"].fillna(float(light_median))
    if include_missing_indicator:
        prepared[MISSING_INDICATOR] = missing.astype(int)
    return prepared


def augment_training_frame(
    frame: pd.DataFrame,
    *,
    scenarios: Sequence[str],
    reference: Any,
    linear_bias_final_std: float,
    out_of_range_std_multiplier: float,
    augmentation_ratio: float = 0.25,
) -> pd.DataFrame:
    """Add an equally allocated fault sample to the unchanged training rows."""

    added_rows = int(round(len(frame) * float(augmentation_ratio)))
    if added_rows < len(scenarios):
        raise ValueError("Training frame is too short for all augmentation faults.")
    ordered = frame.reset_index(drop=True)
    source_boundaries = np.linspace(0, len(ordered), len(scenarios) + 1, dtype=int)
    added_boundaries = np.linspace(0, added_rows, len(scenarios) + 1, dtype=int)
    fault_blocks = []
    for index, scenario in enumerate(scenarios):
        length = int(added_boundaries[index + 1] - added_boundaries[index])
        source_start = int(source_boundaries[index])
        source_end = int(source_boundaries[index + 1])
        available = ordered.iloc[source_start:source_end]
        positions = np.linspace(0, len(available) - 1, length, dtype=int)
        block = available.iloc[positions].reset_index(drop=True)
        block, _ = inject_light_fault(
            block,
            scenario=str(scenario),
            start=0,
            length=length,
            reference=reference,
            linear_bias_final_std=linear_bias_final_std,
            out_of_range_std_multiplier=out_of_range_std_multiplier,
        )
        fault_blocks.append(block)
    return pd.concat([ordered, *fault_blocks], ignore_index=True)


def _whole_period_fault(
    frame: pd.DataFrame,
    scenario: str,
    reference: Any,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    modified, _ = inject_light_fault(
        frame.reset_index(drop=True),
        scenario=scenario,
        start=0,
        length=len(frame),
        reference=reference,
        linear_bias_final_std=float(config["linear_bias_final_std"]),
        out_of_range_std_multiplier=float(config["out_of_range_std_multiplier"]),
    )
    return modified


def _fit(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    *,
    light_median: float,
    include_missing_indicator: bool,
) -> BaseEstimator:
    model = clone(estimator)
    features = prepare_model_input(
        frame,
        light_median=light_median,
        include_missing_indicator=include_missing_indicator,
    )
    with threadpool_limits(limits=1):
        model.fit(features, frame[TARGET_COLUMN])
    return model


def _predict(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    *,
    light_median: float,
    include_missing_indicator: bool,
) -> np.ndarray:
    features = prepare_model_input(
        frame,
        light_median=light_median,
        include_missing_indicator=include_missing_indicator,
    )
    return positive_class_probability(estimator, features)


def cross_validate_fault_aware(
    training: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one model per representation using training folds only."""

    from sklearn.model_selection import TimeSeriesSplit

    ordered = training.sort_values("date").reset_index(drop=True)
    estimators = build_baseline_estimators(int(config["random_seed"]))
    unknown = set(config["candidate_models"]) - set(estimators)
    if unknown:
        raise ValueError(f"Unknown candidate models: {sorted(unknown)}.")
    # These fixed parameters are needed only to derive common training Light
    # references; their detector thresholds do not influence model selection.
    reference_parameters = DetectorParameters(20, 0.0, 0.001, 0.999, 1.0)
    rows: list[dict[str, Any]] = []
    clean_reference_scores: list[float] = []
    splitter = TimeSeriesSplit(n_splits=int(config["cv_splits"]))

    for fold, (development_indices, validation_indices) in enumerate(
        splitter.split(ordered), start=1
    ):
        development = ordered.iloc[development_indices].reset_index(drop=True)
        validation = ordered.iloc[validation_indices].reset_index(drop=True)
        reference = derive_light_reference(
            development,
            reference_parameters,
            {"low": 0.05, "high": 0.95},
        )
        conditions = {"clean": validation}
        conditions.update(
            {
                str(scenario): _whole_period_fault(
                    validation, str(scenario), reference, config
                )
                for scenario in config["fault_scenarios"]
            }
        )
        reference_model_name = str(config["clean_reference_model"])
        if reference_model_name not in estimators:
            raise ValueError(
                f"Unknown clean reference model: {reference_model_name!r}."
            )
        clean_reference = _fit(
            estimators[reference_model_name],
            development,
            light_median=reference.median,
            include_missing_indicator=False,
        )
        clean_reference_probability = _predict(
            clean_reference,
            validation,
            light_median=reference.median,
            include_missing_indicator=False,
        )
        clean_reference_scores.append(
            float(
                classification_metrics(
                    validation[TARGET_COLUMN],
                    clean_reference_probability,
                    threshold=DEFAULT_THRESHOLD,
                )["f1"]
            )
        )
        for representation in config["representations"]:
            indicator = representation == "fault_aware_missing_indicator"
            for augmentation_ratio in config["augmentation_ratios"]:
                augmented = augment_training_frame(
                    development,
                    scenarios=config["fault_scenarios"],
                    reference=reference,
                    linear_bias_final_std=float(config["linear_bias_final_std"]),
                    out_of_range_std_multiplier=float(
                        config["out_of_range_std_multiplier"]
                    ),
                    augmentation_ratio=float(augmentation_ratio),
                )
                for model_name in config["candidate_models"]:
                    fitted = _fit(
                        estimators[str(model_name)],
                        augmented,
                        light_median=reference.median,
                        include_missing_indicator=indicator,
                    )
                    for condition, evaluation in conditions.items():
                        probability = _predict(
                            fitted,
                            evaluation,
                            light_median=reference.median,
                            include_missing_indicator=indicator,
                        )
                        metrics = classification_metrics(
                            validation[TARGET_COLUMN],
                            probability,
                            threshold=DEFAULT_THRESHOLD,
                        )
                        rows.append(
                            {
                                "fold": fold,
                                "representation": representation,
                                "augmentation_ratio": float(augmentation_ratio),
                                "model": model_name,
                                "condition": condition,
                                "is_clean": condition == "clean",
                                **metrics,
                            }
                        )

    details = pd.DataFrame(rows)
    grouped = details.groupby(
        ["representation", "augmentation_ratio", "model"], as_index=False
    )
    summaries = []
    for (representation, augmentation_ratio, model), candidate in grouped:
        clean = candidate.loc[candidate["is_clean"], "f1"]
        fault = candidate.loc[~candidate["is_clean"], "f1"]
        summaries.append(
            {
                "representation": representation,
                "augmentation_ratio": augmentation_ratio,
                "model": model,
                "clean_f1_mean": float(clean.mean()),
                "clean_f1_std": float(clean.std()),
                "fault_f1_mean": float(fault.mean()),
                "fault_f1_std": float(fault.std()),
                "selection_score": float((clean.mean() + fault.mean()) / 2),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["eligible"] = False
    summary["selected"] = False
    summary["guardrail_satisfied"] = False
    clean_reference_f1_mean = float(np.mean(clean_reference_scores))
    summary["clean_reference_f1_mean"] = clean_reference_f1_mean
    clean_floor = clean_reference_f1_mean - float(
        config["maximum_clean_f1_drop"]
    )
    for representation, indices in summary.groupby("representation").groups.items():
        candidates = summary.loc[indices]
        eligible = candidates["clean_f1_mean"] >= clean_floor
        summary.loc[candidates.index, "eligible"] = eligible
        if eligible.any():
            ranked = candidates.loc[eligible].sort_values(
                [
                    "selection_score",
                    "clean_f1_mean",
                    "augmentation_ratio",
                    "model",
                ],
                ascending=[False, False, True, True],
            )
            summary.loc[ranked.index[0], "guardrail_satisfied"] = True
        else:
            ranked = candidates.sort_values(
                ["clean_f1_mean", "selection_score", "augmentation_ratio", "model"],
                ascending=[False, False, True, True],
            )
        summary.loc[ranked.index[0], "selected"] = True
    summary = summary.sort_values(
        ["representation", "selected", "selection_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return details, summary


def _load_selected_bundle(
    directory: Path, selected: pd.DataFrame, feature_set: str
) -> dict[str, Any]:
    row = selected.loc[selected["feature_set"] == feature_set]
    if len(row) != 1:
        raise ValueError(f"Unknown selected feature set: {feature_set!r}.")
    model_name = str(row.iloc[0]["selected_model"])
    return joblib.load(directory / f"{feature_set}__{model_name}.joblib")


def _bundle_probability(
    bundle: Mapping[str, Any], frame: pd.DataFrame, light_median: float | None = None
) -> np.ndarray:
    features = list(bundle["feature_columns"])
    values = frame[features].copy()
    if light_median is not None and "Light" in values:
        values["Light"] = values["Light"].fillna(light_median)
    return positive_class_probability(bundle["estimator"], values)


def run_fault_aware_evaluation(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    fault_detection_config_path: Path | str = DEFAULT_FAULT_DETECTION_CONFIG_PATH,
    sensor_budget_dir: Path | str = DEFAULT_SENSOR_BUDGET_DIR,
    fault_detection_dir: Path | str = DEFAULT_FAULT_DETECTION_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Train selected fault-aware models and compare held-out strategies."""

    validate_source_checksums(raw_dir, checksum_path)
    frames = load_source_splits(raw_dir)
    validate_source_splits(frames)
    config_source = Path(config_path)
    config = load_fault_aware_config(config_source)
    training = frames["train"].sort_values("date").reset_index(drop=True)
    cv_details, cv_summary = cross_validate_fault_aware(training, config)

    detector_metadata_path = Path(fault_detection_dir) / "metadata.json"
    if not detector_metadata_path.exists():
        raise FileNotFoundError(
            "Run `python -m sensorbudget.robustness.fault_detection` first."
        )
    detector_metadata = json.loads(detector_metadata_path.read_text("utf-8"))
    detector_config = json.loads(
        Path(fault_detection_config_path).read_text(encoding="utf-8")
    )
    selected_detector = detector_metadata["selected_detector"]
    detector_parameters = DetectorParameters(
        stuck_window=int(selected_detector["stuck_window"]),
        stuck_tolerance_lux=float(selected_detector["stuck_tolerance_lux"]),
        range_low_quantile=float(selected_detector["range_low_quantile"]),
        range_high_quantile=float(selected_detector["range_high_quantile"]),
        abrupt_change_quantile=float(selected_detector["abrupt_change_quantile"]),
    )
    reference = derive_light_reference(
        training,
        detector_parameters,
        detector_config["stuck_reference_quantiles"],
    )
    estimators = build_baseline_estimators(int(config["random_seed"]))
    fitted_models: dict[str, dict[str, Any]] = {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for row in cv_summary.loc[cv_summary["selected"]].itertuples(index=False):
        indicator = row.representation == "fault_aware_missing_indicator"
        augmented = augment_training_frame(
            training,
            scenarios=config["fault_scenarios"],
            reference=reference,
            linear_bias_final_std=float(config["linear_bias_final_std"]),
            out_of_range_std_multiplier=float(
                config["out_of_range_std_multiplier"]
            ),
            augmentation_ratio=float(row.augmentation_ratio),
        )
        fitted = _fit(
            estimators[row.model],
            augmented,
            light_median=reference.median,
            include_missing_indicator=indicator,
        )
        bundle = {
            "estimator": fitted,
            "representation": row.representation,
            "model_name": row.model,
            "augmentation_ratio": float(row.augmentation_ratio),
            "feature_columns": PRIMARY_COLUMNS
            + ([MISSING_INDICATOR] if indicator else []),
            "light_median": reference.median,
            "threshold": DEFAULT_THRESHOLD,
        }
        fitted_models[row.representation] = bundle
        joblib.dump(bundle, output / f"{row.representation}__{row.model}.joblib")

    budget_directory = Path(sensor_budget_dir)
    selected_models = pd.read_csv(budget_directory / "selected_models.csv")
    primary_bundle = _load_selected_bundle(
        budget_directory, selected_models, str(config["primary_feature_set"])
    )
    fallback_bundle = _load_selected_bundle(
        budget_directory, selected_models, str(config["fallback_feature_set"])
    )
    threshold = float(primary_bundle["threshold"])
    rows: list[dict[str, Any]] = []
    base_seed = int(config["random_seed"])
    margin = detector_parameters.stuck_window + 2

    def record(
        split: str,
        scenario: str,
        length: int,
        repeat: int,
        strategy: str,
        probability: np.ndarray,
        truth: pd.Series | None,
        target: pd.Series,
    ) -> None:
        overall = classification_metrics(target, probability, threshold=threshold)
        fault_window = (
            classification_metrics(
                target.loc[truth], probability[truth.to_numpy()], threshold=threshold
            )
            if truth is not None and truth.any()
            else {key: float("nan") for key in overall}
        )
        rows.append(
            {
                "split": split,
                "scenario": scenario,
                "episode_length": length,
                "repeat": repeat,
                "strategy": strategy,
                **overall,
                **{f"fault_window_{key}": value for key, value in fault_window.items()},
            }
        )

    for split in config["heldout_splits"]:
        clean = frames[str(split)].sort_values("date").reset_index(drop=True)
        primary_clean = _bundle_probability(
            primary_bundle, clean, light_median=reference.median
        )
        fallback_clean = _bundle_probability(fallback_bundle, clean)
        detector_clean = detect_light_faults(
            clean["Light"], parameters=detector_parameters, reference=reference
        )["detected_light_fault"].to_numpy()
        clean_probabilities = {
            "primary_only": primary_clean,
            "detector_routing": route_probabilities(
                primary_clean, fallback_clean, detector_clean
            ),
        }
        for representation, bundle in fitted_models.items():
            clean_probabilities[representation] = _predict(
                bundle["estimator"],
                clean,
                light_median=reference.median,
                include_missing_indicator=MISSING_INDICATOR
                in bundle["feature_columns"],
            )
        for strategy, probability in clean_probabilities.items():
            record(
                str(split),
                "clean",
                0,
                0,
                strategy,
                probability,
                None,
                clean[TARGET_COLUMN],
            )

        for scenario in config["fault_scenarios"]:
            for length in config["heldout_episode_lengths"]:
                for repeat in range(int(config["heldout_repeats"])):
                    seed = stable_random_seed(
                        base_seed, "heldout", split, scenario, length, repeat
                    )
                    start = choose_episode_start(
                        len(clean), int(length), margin=margin,
                        rng=np.random.default_rng(seed),
                    )
                    modified, truth = inject_light_fault(
                        clean,
                        scenario=str(scenario),
                        start=start,
                        length=int(length),
                        reference=reference,
                        linear_bias_final_std=float(config["linear_bias_final_std"]),
                        out_of_range_std_multiplier=float(
                            config["out_of_range_std_multiplier"]
                        ),
                    )
                    primary = _bundle_probability(
                        primary_bundle, modified, light_median=reference.median
                    )
                    detected = detect_light_faults(
                        modified["Light"],
                        parameters=detector_parameters,
                        reference=reference,
                    )["detected_light_fault"].to_numpy()
                    probabilities = {
                        "primary_only": primary,
                        "detector_routing": route_probabilities(
                            primary, fallback_clean, detected
                        ),
                    }
                    for representation, bundle in fitted_models.items():
                        probabilities[representation] = _predict(
                            bundle["estimator"],
                            modified,
                            light_median=reference.median,
                            include_missing_indicator=MISSING_INDICATOR
                            in bundle["feature_columns"],
                        )
                    for strategy, probability in probabilities.items():
                        record(
                            str(split), str(scenario), int(length), repeat + 1,
                            strategy, probability, truth, clean[TARGET_COLUMN]
                        )

    heldout = pd.DataFrame(rows)
    write_table(output / "cv_fold_metrics.csv", cv_details)
    write_table(output / "cv_summary.csv", cv_summary)
    write_table(output / "heldout_metrics.csv", heldout)
    metadata = {
        "config_path": str(config_source),
        "config_sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
        "selection_source": "chronological folds of the supplied training period only",
        "selection_rule": config["selection_rule"],
        "selected_models": {
            row.representation: {
                "model": row.model,
                "augmentation_ratio": float(row.augmentation_ratio),
                "guardrail_satisfied": bool(row.guardrail_satisfied),
            }
            for row in cv_summary.loc[cv_summary["selected"]].itertuples(index=False)
        },
        "training_rows_clean": len(training),
        "light_training_median": reference.median,
        "heldout_data_used_for_selection": False,
    }
    write_json(output / "metadata.json", metadata)
    return {
        "cv_fold_metrics": cv_details,
        "cv_summary": cv_summary,
        "heldout_metrics": heldout,
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run fault-aware training and held-out evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUM_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_fault_aware_evaluation(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print("Selected fault-aware models:")
    print(json.dumps(result["metadata"]["selected_models"], indent=2))
    print("\nHeld-out mean F1:")
    print(
        result["heldout_metrics"]
        .groupby(["split", "strategy"], as_index=False)["f1"]
        .mean()
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
