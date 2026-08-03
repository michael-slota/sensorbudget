"""Export fallback, detector-routing, and fault-aware mitigation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_OUTPUT_PATH = Path("site/data/fault-mitigation.json")
FEATURE_LABELS = {
    "temperature__co2": "Temperature + CO2",
    "co2": "CO2",
    "temperature": "Temperature",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Histogram gradient boosting",
}
SPLIT_LABELS = {"test_1": "Test 1", "test_2": "Test 2"}
STRATEGY_LABELS = {
    "primary_only": "Primary only",
    "fallback_only": "Fallback only",
    "oracle_routing": "Oracle routing",
    "detector_routing": "Detector routing",
    "fault_aware": "Fault-aware (diagnostic)",
    "fault_aware_missing_indicator": "Fault-aware + missing indicator",
}
SCENARIO_LABELS = {
    "missing": "Missing Light",
    "out_of_range_high": "Above training range",
    "stuck_current": "Frozen at current value",
    "stuck_high": "Fixed high",
    "stuck_low": "Fixed low / darkness",
    "linear_bias_negative": "Negative linear bias",
    "linear_bias_positive": "Positive linear bias",
}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def build_mitigation_payload(
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> dict[str, object]:
    """Build compact tables spanning the three Phase 5 mitigation stages."""

    fallback_dir = models_dir / "fallback_mitigation"
    detector_dir = models_dir / "fault_detection"
    fault_aware_dir = models_dir / "fault_aware"
    candidates = pd.read_csv(fallback_dir / "fallback_candidates.csv")
    fallback = pd.read_csv(fallback_dir / "fallback_metrics.csv")
    detection = pd.read_csv(detector_dir / "heldout_detection_metrics.csv")
    routing = pd.read_csv(detector_dir / "heldout_routing_metrics.csv")
    fault_cv = pd.read_csv(fault_aware_dir / "cv_summary.csv")
    fault_heldout = pd.read_csv(fault_aware_dir / "heldout_metrics.csv")

    candidates["feature_label"] = candidates["feature_set"].map(FEATURE_LABELS)
    candidates["model_label"] = candidates["selected_model"].map(MODEL_LABELS)

    scenario_labels = {
        "median_imputation": "Complete Light loss",
        "stuck_low": "Light fixed low",
        "stuck_high": "Light fixed high",
        "unoccupied_lit": "Unoccupied but lit",
        "occupied_dark": "Occupied but dark",
    }
    oracle = fallback.loc[
        fallback["scenario"].isin(scenario_labels)
        & fallback["strategy"].isin(
            ["primary_under_fault", "oracle_gated_fallback"]
        ),
        ["split", "scenario", "strategy", "f1"],
    ].copy()
    oracle["split_label"] = oracle["split"].map(SPLIT_LABELS)
    oracle["scenario_label"] = oracle["scenario"].map(scenario_labels)
    oracle["strategy_label"] = oracle["strategy"].map(
        {
            "primary_under_fault": "Primary under fault",
            "oracle_gated_fallback": "Oracle fallback",
        }
    )

    detection_summary = (
        detection.loc[detection["scenario"].ne("clean")]
        .groupby(["split", "scenario"], observed=True)
        .agg(
            detection_precision=("detection_precision", "mean"),
            detection_recall=("detection_recall", "mean"),
            detection_f1=("detection_f1", "mean"),
            false_positive_rate=("false_positive_rate", "mean"),
            detection_delay_rows=("detection_delay_rows", "mean"),
        )
        .reset_index()
    )
    detection_summary["split_label"] = detection_summary["split"].map(SPLIT_LABELS)
    detection_summary["scenario_label"] = detection_summary["scenario"].map(
        SCENARIO_LABELS
    )

    routing_summary = (
        routing.groupby(["split", "strategy"], observed=True)["f1"]
        .mean()
        .reset_index(name="mean_f1")
    )
    routing_summary["split_label"] = routing_summary["split"].map(SPLIT_LABELS)
    routing_summary["strategy_label"] = routing_summary["strategy"].map(
        STRATEGY_LABELS
    )

    selected_cv = fault_cv.loc[fault_cv["selected"].astype(bool)].copy()
    selected_cv["representation_label"] = selected_cv["representation"].map(
        {
            "fault_aware": "Fault-aware (diagnostic)",
            "fault_aware_missing_indicator": "Fault-aware + missing indicator",
        }
    )
    selected_cv["model_label"] = selected_cv["model"].map(MODEL_LABELS)

    clean = fault_heldout.loc[fault_heldout["scenario"].eq("clean")].drop_duplicates(
        ["split", "strategy"]
    )
    clean = clean.copy()
    clean["split_label"] = clean["split"].map(SPLIT_LABELS)
    clean["strategy_label"] = clean["strategy"].map(STRATEGY_LABELS)

    strategy_average = (
        fault_heldout.groupby(["split", "strategy"], observed=True)["f1"]
        .mean()
        .reset_index(name="mean_f1")
    )
    strategy_average["split_label"] = strategy_average["split"].map(SPLIT_LABELS)
    strategy_average["strategy_label"] = strategy_average["strategy"].map(
        STRATEGY_LABELS
    )

    return {
        "metadata": {
            "primary": "Temperature + Light + CO2 logistic regression",
            "fallback": "Temperature + CO2 logistic regression",
            "threshold": 0.5,
        },
        "fallback_candidates": _records(
            candidates[
                [
                    "feature_set",
                    "feature_label",
                    "model_label",
                    "cv_f1_mean",
                    "selected_fallback",
                ]
            ]
        ),
        "oracle_recovery": _records(oracle),
        "detector_quality": _records(detection_summary),
        "routing_average": _records(routing_summary),
        "fault_aware_validation": _records(
            selected_cv[
                [
                    "representation_label",
                    "augmentation_ratio",
                    "model_label",
                    "clean_f1_mean",
                    "fault_f1_mean",
                    "guardrail_satisfied",
                    "clean_reference_f1_mean",
                ]
            ]
        ),
        "clean_heldout": _records(
            clean[["split", "split_label", "strategy", "strategy_label", "f1"]]
        ),
        "strategy_average": _records(strategy_average),
    }


def export_mitigation_data(
    models_dir: Path = DEFAULT_MODELS_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the combined mitigation dashboard aggregate as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_mitigation_payload(models_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output = export_mitigation_data(args.models_dir, args.output)
    print(f"Wrote fault-mitigation dashboard data to {output}")


if __name__ == "__main__":
    main()
