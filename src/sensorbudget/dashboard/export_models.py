"""Export aggregate model-performance evidence for the static dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

DEFAULT_BASELINE_DIR = Path("models/baseline")
DEFAULT_SENSOR_BUDGET_DIR = Path("models/sensor_budget")
DEFAULT_OUTPUT_PATH = Path("site/data/model-performance.json")
PRIMARY_FEATURE_SET = "temperature__light__co2"

MODEL_LABELS = {
    "dummy_prior": "Dummy prior",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}
FEATURE_LABELS = {
    "all_sensors": "All-sensor baseline",
    "no_light": "No-Light baseline",
    PRIMARY_FEATURE_SET: "Three-sensor candidate",
}
SPLIT_LABELS = {"test_1": "Test 1", "test_2": "Test 2"}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame into JSON-safe records."""

    return json.loads(frame.to_json(orient="records"))


def _downsample_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    maximum_points: int = 120,
) -> list[dict[str, float]]:
    """Keep an evenly spaced, hoverable subset of a threshold curve."""

    available = len(thresholds)
    if available <= maximum_points:
        indices = np.arange(available)
    else:
        indices = np.unique(np.linspace(0, available - 1, maximum_points).astype(int))
    return [
        {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "threshold": float(thresholds[index]),
        }
        for index in indices
    ]


def build_model_performance_payload(
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    sensor_budget_dir: Path = DEFAULT_SENSOR_BUDGET_DIR,
) -> dict[str, object]:
    """Build compact dashboard tables from reproducible modeling artifacts."""

    baseline_cv = pd.read_csv(baseline_dir / "cv_summary.csv")
    baseline_heldout = pd.read_csv(baseline_dir / "heldout_metrics.csv")
    baseline_predictions = pd.read_csv(baseline_dir / "heldout_predictions.csv")
    sensor_heldout = pd.read_csv(sensor_budget_dir / "heldout_metrics.csv")
    sensor_predictions = pd.read_csv(sensor_budget_dir / "heldout_predictions.csv")

    cv = baseline_cv.loc[
        :,
        ["feature_set", "model", "f1_mean", "f1_std", "precision_mean", "recall_mean"],
    ].copy()
    cv["feature_label"] = cv["feature_set"].map(FEATURE_LABELS)
    cv["model_label"] = cv["model"].map(MODEL_LABELS)

    heldout = baseline_heldout.loc[
        :,
        [
            "feature_set",
            "model",
            "split",
            "f1",
            "precision",
            "recall",
            "false_positive",
            "false_negative",
        ],
    ].copy()
    heldout["feature_label"] = heldout["feature_set"].map(FEATURE_LABELS)
    heldout["split_label"] = heldout["split"].map(SPLIT_LABELS)

    primary = sensor_heldout.loc[
        sensor_heldout["feature_set"].eq(PRIMARY_FEATURE_SET),
        [
            "feature_set",
            "model",
            "split",
            "f1",
            "precision",
            "recall",
            "false_positive",
            "false_negative",
            "true_positive",
            "true_negative",
        ],
    ].copy()
    primary["feature_label"] = primary["feature_set"].map(FEATURE_LABELS)
    primary["split_label"] = primary["split"].map(SPLIT_LABELS)
    primary["model_label"] = primary["model"].map(MODEL_LABELS)

    comparison = pd.concat(
        [
            baseline_heldout.loc[
                baseline_heldout["feature_set"].eq("all_sensors"),
                ["feature_set", "model", "split", "f1", "precision", "recall"],
            ],
            primary.loc[
                :, ["feature_set", "model", "split", "f1", "precision", "recall"]
            ],
        ],
        ignore_index=True,
    )
    comparison["feature_label"] = comparison["feature_set"].map(FEATURE_LABELS)
    comparison["split_label"] = comparison["split"].map(SPLIT_LABELS)
    comparison["model_label"] = comparison["model"].map(MODEL_LABELS)

    curves = []
    prediction_sets = [
        (
            baseline_predictions.loc[
                baseline_predictions["feature_set"].eq("all_sensors")
            ],
            "All-sensor baseline",
        ),
        (
            sensor_predictions.loc[
                sensor_predictions["feature_set"].eq(PRIMARY_FEATURE_SET)
            ],
            "Three-sensor candidate",
        ),
    ]
    for predictions, candidate_label in prediction_sets:
        for split, frame in predictions.groupby("source_split", sort=False):
            precision, recall, thresholds = precision_recall_curve(
                frame["Occupancy"], frame["probability_occupied"]
            )
            curves.append(
                {
                    "candidate": candidate_label,
                    "split": split,
                    "split_label": SPLIT_LABELS[split],
                    "points": _downsample_curve(precision, recall, thresholds),
                }
            )

    return {
        "metadata": {
            "baseline_candidate": "All-sensor histogram gradient boosting",
            "research_candidate": "Temperature + Light + CO2 logistic regression",
            "threshold": 0.5,
            "selection_metric": "Mean chronological-validation F1",
        },
        "cv_summary": _records(cv),
        "baseline_heldout": _records(heldout),
        "candidate_comparison": _records(comparison),
        "primary_confusion": _records(primary),
        "precision_recall_curves": curves,
    }


def export_model_performance_data(
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    sensor_budget_dir: Path = DEFAULT_SENSOR_BUDGET_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the model-performance dashboard aggregate as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_model_performance_payload(baseline_dir, sensor_budget_dir),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Parse CLI arguments and export model-performance dashboard data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--sensor-budget-dir", type=Path, default=DEFAULT_SENSOR_BUDGET_DIR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output = export_model_performance_data(
        args.baseline_dir, args.sensor_budget_dir, args.output
    )
    print(f"Wrote model-performance dashboard data to {output}")


if __name__ == "__main__":
    main()
