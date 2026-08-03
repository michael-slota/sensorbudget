from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensorbudget.modeling.decision_explainability import (
    add_prediction_outcomes,
    calibration_table,
    evaluate_thresholds,
    load_decision_config,
    occupancy_phase_table,
    select_operating_thresholds,
    threshold_grid,
)


def test_threshold_cost_uses_configured_false_positive_and_negative_costs() -> None:
    curves = evaluate_thresholds(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.7, 0.2]),
        [0.5],
        {
            "example": {
                "label": "Example",
                "false_positive_cost": 2.0,
                "false_negative_cost": 5.0,
            }
        },
    )

    assert curves.loc[0, "false_positive"] == 1
    assert curves.loc[0, "false_negative"] == 1
    assert curves.loc[0, "cost_per_1000_rows"] == pytest.approx(1750.0)


def test_threshold_selection_minimizes_cost_then_maximizes_f1() -> None:
    curves = pd.DataFrame(
        {
            "scenario": ["a", "a", "a"],
            "threshold": [0.3, 0.4, 0.5],
            "cost_per_1000_rows": [10.0, 10.0, 20.0],
            "f1": [0.7, 0.8, 0.9],
        }
    )

    selected = select_operating_thresholds(curves)

    assert selected.loc[0, "threshold"] == pytest.approx(0.4)


def test_calibration_table_ece_is_zero_for_matching_bin_means() -> None:
    table, summary = calibration_table(
        np.array([0, 1]), np.array([0.0, 1.0]), bins=2, dataset="sample"
    )

    assert table["row_count"].sum() == 2
    assert summary["brier_score"] == pytest.approx(0.0)
    assert summary["expected_calibration_error"] == pytest.approx(0.0)


def test_prediction_outcomes_assign_all_confusion_categories() -> None:
    frame = pd.DataFrame({"Occupancy": [1, 0, 0, 1]})
    result = add_prediction_outcomes(
        frame, np.array([0.9, 0.1, 0.8, 0.2]), threshold=0.5
    )

    assert result["outcome"].tolist() == [
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ]


def test_occupancy_phase_table_tracks_recall_after_onset() -> None:
    frame = pd.DataFrame(
        {
            "source_split": ["test"] * 7,
            "date": pd.date_range("2026-01-01", periods=7, freq="min"),
            "Occupancy": [0, 1, 1, 1, 1, 1, 1],
            "predicted_occupancy": [0, 0, 0, 1, 1, 1, 1],
        }
    )

    result = occupancy_phase_table(frame)
    onset = result.loc[result["occupancy_phase"] == "0–2 min"].iloc[0]

    assert onset["occupied_rows"] == 3
    assert onset["false_negative"] == 2
    assert onset["recall"] == pytest.approx(1 / 3)


def test_project_config_builds_expected_threshold_grid() -> None:
    config = load_decision_config("configs/decision_explainability.json")
    values = threshold_grid(config)

    assert values[0] == pytest.approx(0.01)
    assert values[-1] == pytest.approx(0.99)
    assert len(values) == 99
