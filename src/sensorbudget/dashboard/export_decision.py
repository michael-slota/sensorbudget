"""Export Phase 6 decision-threshold and explainability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_ARTIFACT_DIR = Path("models/decision_explainability")
DEFAULT_OUTPUT_PATH = Path("site/data/decision-explainability.json")
DATASET_LABELS = {
    "chronological_cv": "Chronological validation",
    "test_1": "Test 1",
    "test_2": "Test 2",
}
OUTCOME_LABELS = {
    "true_positive": "True positive",
    "true_negative": "True negative",
    "false_positive": "False positive",
    "false_negative": "False negative",
}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def build_decision_payload(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, object]:
    """Build compact tables for the final dashboard."""

    selected = pd.read_csv(artifact_dir / "selected_thresholds.csv")
    curves = pd.read_csv(artifact_dir / "threshold_curves.csv")
    heldout = pd.read_csv(artifact_dir / "heldout_metrics.csv")
    calibration = pd.read_csv(artifact_dir / "calibration_summary.csv")
    bins = pd.read_csv(artifact_dir / "calibration_bins.csv")
    coefficients = pd.read_csv(artifact_dir / "global_coefficients.csv")
    explanations = pd.read_csv(artifact_dir / "representative_explanations.csv")
    transitions = pd.read_csv(artifact_dir / "transition_metrics.csv")

    heldout_equal = heldout.loc[heldout["scenario"].eq("equal_cost")].copy()
    heldout_equal["split_label"] = heldout_equal["split"].map(DATASET_LABELS)
    heldout_equal["threshold_label"] = heldout_equal["threshold_source"].map(
        {"default_0.5": "Default 0.50", "validation_selected": "Selected 0.86"}
    )

    calibration["dataset_label"] = calibration["dataset"].map(DATASET_LABELS)
    bins["dataset_label"] = bins["dataset"].map(DATASET_LABELS)

    closest = (
        explanations.sort_values("distance_from_threshold")
        .groupby(["source_split", "outcome"], observed=True)
        .head(1)
        .copy()
    )
    closest["split_label"] = closest["source_split"].map(DATASET_LABELS)
    closest["outcome_label"] = closest["outcome"].map(OUTCOME_LABELS)
    closest["row_label"] = closest["split_label"] + " · " + closest["outcome_label"]

    transitions["split_label"] = transitions["split"].map(DATASET_LABELS)

    return {
        "metadata": {
            "candidate": "Temperature + Light + CO2 logistic regression",
            "default_threshold": 0.5,
            "reference_selected_threshold": 0.86,
            "threshold_selection_data": "Chronological validation only",
        },
        "selected_thresholds": _records(selected),
        "threshold_curves": _records(curves),
        "heldout_equal_cost": _records(heldout_equal),
        "calibration_summary": _records(calibration),
        "calibration_bins": _records(bins),
        "global_coefficients": _records(coefficients),
        "local_explanations": _records(closest),
        "transition_metrics": _records(transitions),
    }


def export_decision_data(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the decision dashboard aggregate as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_decision_payload(artifact_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output = export_decision_data(args.artifact_dir, args.output)
    print(f"Wrote decision-and-explainability dashboard data to {output}")


if __name__ == "__main__":
    main()
