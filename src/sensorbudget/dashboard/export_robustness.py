"""Export Phase 5 fault-injection evidence for the static dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_ARTIFACT_DIR = Path("models/robustness")
DEFAULT_OUTPUT_PATH = Path("site/data/robustness.json")
PRIMARY_FEATURE_SET = "temperature__light__co2"

FEATURE_LABELS = {
    "light": "Light",
    "humidity__light": "Humidity + Light",
    PRIMARY_FEATURE_SET: "Temperature + Light + CO2",
}
SPLIT_LABELS = {"test_1": "Test 1", "test_2": "Test 2"}
SENSOR_LABELS = {
    "Temperature": "Temperature",
    "Humidity": "Humidity",
    "Light": "Light",
    "CO2": "CO2",
}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-safe records."""

    return json.loads(frame.to_json(orient="records"))


def _summarize_repetitions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated fault draws while retaining scenario identity."""

    group_columns = [
        "feature_set",
        "split",
        "scenario_group",
        "scenario",
        "severity",
        "sensor",
    ]
    return (
        frame.groupby(group_columns, dropna=False, observed=True)["f1"]
        .agg(f1_mean="mean", f1_std="std", repetitions="size")
        .reset_index()
        .fillna({"f1_std": 0.0})
    )


def _add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach consistent public labels to aggregate rows."""

    result = frame.copy()
    result["feature_label"] = result["feature_set"].map(FEATURE_LABELS)
    result["split_label"] = result["split"].map(SPLIT_LABELS)
    result["sensor_label"] = result["sensor"].map(SENSOR_LABELS).fillna(
        result["sensor"]
    )
    return result


def build_robustness_payload(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, object]:
    """Build compact robustness tables from the reproducible fault artifacts."""

    metrics = pd.read_csv(artifact_dir / "robustness_metrics.csv")
    summary = _add_labels(_summarize_repetitions(metrics))

    baseline = summary.loc[summary["scenario_group"].eq("baseline")].copy()
    baseline["display_scenario"] = "Clean"

    policy = summary.loc[summary["scenario_group"].eq("light_policy")].copy()
    policy["display_scenario"] = policy["scenario"].map(
        {"occupied_dark": "Occupied but dark", "unoccupied_lit": "Unoccupied but lit"}
    )

    light_loss = summary.loc[
        summary["scenario_group"].eq("complete_loss")
        & summary["sensor"].eq("Light")
    ].copy()
    light_loss["display_scenario"] = "Complete Light loss"
    overview = pd.concat([baseline, policy, light_loss], ignore_index=True)

    clean_origin = baseline.copy()
    clean_origin["severity_numeric"] = 0.0

    missing = summary.loc[summary["scenario_group"].eq("random_missing")].copy()
    missing["severity_numeric"] = pd.to_numeric(missing["severity"])
    missing = pd.concat([clean_origin, missing], ignore_index=True)

    noise = summary.loc[summary["scenario_group"].eq("gaussian_noise")].copy()
    noise["severity_numeric"] = pd.to_numeric(noise["severity"])
    noise = pd.concat([clean_origin, noise], ignore_index=True)

    primary_loss = summary.loc[
        summary["scenario_group"].eq("complete_loss")
        & summary["feature_set"].eq(PRIMARY_FEATURE_SET)
    ].copy()

    primary_stuck = summary.loc[
        summary["scenario_group"].eq("stuck_sensor")
        & summary["feature_set"].eq(PRIMARY_FEATURE_SET)
    ].copy()
    primary_stuck["stuck_position"] = primary_stuck["scenario"].map(
        {"stuck_low": "Fixed low", "stuck_high": "Fixed high"}
    )

    light_drift = summary.loc[
        summary["scenario_group"].eq("gradual_drift")
        & summary["sensor"].eq("Light")
    ].copy()
    light_drift["severity_numeric"] = pd.to_numeric(light_drift["severity"])
    light_drift = pd.concat([clean_origin, light_drift], ignore_index=True)

    columns = [
        "feature_set",
        "feature_label",
        "split",
        "split_label",
        "scenario",
        "severity",
        "sensor",
        "sensor_label",
        "f1_mean",
        "f1_std",
        "repetitions",
    ]
    overview_columns = [*columns, "display_scenario"]
    severity_columns = [*columns, "severity_numeric"]
    stuck_columns = [*columns, "stuck_position"]

    return {
        "metadata": {
            "configuration_count": 3,
            "heldout_period_count": 2,
            "threshold": 0.5,
            "fault_type": "Synthetic diagnostic perturbations",
            "recommendation_status": "No final sensor recommendation",
        },
        "overview": _records(overview.loc[:, overview_columns]),
        "random_missing": _records(missing.loc[:, severity_columns]),
        "gaussian_noise": _records(noise.loc[:, severity_columns]),
        "primary_sensor_loss": _records(primary_loss.loc[:, columns]),
        "primary_stuck_sensor": _records(primary_stuck.loc[:, stuck_columns]),
        "light_drift": _records(light_drift.loc[:, severity_columns]),
    }


def export_robustness_data(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the robustness dashboard aggregate as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_robustness_payload(artifact_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Parse CLI arguments and export robustness dashboard data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output = export_robustness_data(args.artifact_dir, args.output)
    print(f"Wrote robustness dashboard data to {output}")


if __name__ == "__main__":
    main()
