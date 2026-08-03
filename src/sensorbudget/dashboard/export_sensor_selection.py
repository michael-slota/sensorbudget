"""Export Phase 4 sensor-selection evidence for the static dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_ARTIFACT_DIR = Path("models/sensor_budget")
DEFAULT_OUTPUT_PATH = Path("site/data/sensor-selection.json")

MODEL_LABELS = {
    "dummy_prior": "Dummy prior",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}


def _sensor_set(value: str) -> frozenset[str]:
    """Convert the artifact's comma-separated physical sensors into a set."""

    return frozenset(part.strip() for part in value.split(","))


def _sensor_label(value: str) -> str:
    """Format physical-sensor names for public display."""

    names = {
        "temperature": "Temperature",
        "humidity": "Humidity",
        "light": "Light",
        "co2": "CO2",
    }
    return " + ".join(names[part.strip()] for part in value.split(","))


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame into JSON-safe records."""

    return json.loads(frame.to_json(orient="records"))


def build_sensor_selection_payload(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, object]:
    """Build compact tables for the sensor-selection dashboard."""

    selected = pd.read_csv(artifact_dir / "selected_models.csv")
    heldout = pd.read_csv(artifact_dir / "heldout_metrics.csv")
    sensitivity = pd.read_csv(artifact_dir / "cost_sensitivity.csv")
    frequency = pd.read_csv(artifact_dir / "cost_sensitivity_frequency.csv")

    selected = selected.copy()
    selected["sensor_label"] = selected["physical_sensors"].map(_sensor_label)
    selected["model_label"] = selected["selected_model"].map(MODEL_LABELS)
    selected = selected.sort_values("cv_f1_mean", ascending=False)

    current = sensitivity.loc[
        sensitivity["scenario"].eq("current_assumptions"),
        [
            "feature_set",
            "physical_sensors",
            "selected_model",
            "cv_f1_mean",
            "scenario_cost",
            "is_pareto",
        ],
    ].copy()
    current["sensor_label"] = current["physical_sensors"].map(_sensor_label)
    current["model_label"] = current["selected_model"].map(MODEL_LABELS)

    by_sensor_set = {
        _sensor_set(row.physical_sensors): row
        for row in selected.itertuples(index=False)
    }
    light_additions = []
    for sensors, without_row in by_sensor_set.items():
        if "light" in sensors:
            continue
        with_light = sensors | {"light"}
        if with_light not in by_sensor_set:
            continue
        with_row = by_sensor_set[with_light]
        light_additions.append(
            {
                "base_sensors": _sensor_label(without_row.physical_sensors),
                "with_light_sensors": _sensor_label(with_row.physical_sensors),
                "without_light_f1": float(without_row.cv_f1_mean),
                "with_light_f1": float(with_row.cv_f1_mean),
                "f1_change": float(with_row.cv_f1_mean - without_row.cv_f1_mean),
            }
        )
    light_additions.sort(key=lambda row: row["f1_change"])

    frequency = frequency.copy()
    frequency["sensor_label"] = frequency["physical_sensors"].map(_sensor_label)
    frequency = frequency.sort_values(
        ["frontier_scenarios", "cv_f1_mean"], ascending=[False, False]
    )

    stability = heldout.pivot(
        index=["feature_set", "model", "relative_cost", "physical_sensor_count"],
        columns="split",
        values="f1",
    ).reset_index()
    labels = selected.set_index("feature_set")["sensor_label"]
    stability["sensor_label"] = stability["feature_set"].map(labels)
    stability["model_label"] = stability["model"].map(MODEL_LABELS)

    selected_columns = [
        "feature_set",
        "sensor_label",
        "physical_sensor_count",
        "relative_cost",
        "model_label",
        "cv_f1_mean",
    ]
    current_columns = [
        "feature_set",
        "sensor_label",
        "model_label",
        "cv_f1_mean",
        "scenario_cost",
        "is_pareto",
    ]
    frequency_columns = [
        "feature_set",
        "sensor_label",
        "frontier_scenarios",
        "total_scenarios",
        "cv_f1_mean",
        "frontier_share",
    ]
    stability_columns = [
        "feature_set",
        "sensor_label",
        "model_label",
        "relative_cost",
        "physical_sensor_count",
        "test_1",
        "test_2",
    ]

    return {
        "metadata": {
            "configuration_count": 15,
            "cost_scenario_count": 5,
            "cost_unit": "Illustrative relative cost points",
            "selection_metric": "Mean chronological-validation F1",
            "recommendation_status": "No final sensor recommendation",
        },
        "validation_ranking": _records(selected.loc[:, selected_columns]),
        "current_cost_frontier": _records(current.loc[:, current_columns]),
        "light_additions": light_additions,
        "frontier_frequency": _records(frequency.loc[:, frequency_columns]),
        "heldout_stability": _records(stability.loc[:, stability_columns]),
    }


def export_sensor_selection_data(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the sensor-selection dashboard aggregate as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_sensor_selection_payload(artifact_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Parse CLI arguments and export sensor-selection dashboard data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output = export_sensor_selection_data(args.artifact_dir, args.output)
    print(f"Wrote sensor-selection dashboard data to {output}")


if __name__ == "__main__":
    main()
