from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensorbudget.robustness.fault_detection import (
    DetectorParameters,
    LightReference,
    build_parameter_grid,
    detect_light_faults,
    fault_detection_metrics,
    inject_light_fault,
    route_probabilities,
)

PARAMETERS = DetectorParameters(
    stuck_window=3,
    stuck_tolerance_lux=0.0,
    range_low_quantile=0.0,
    range_high_quantile=1.0,
    abrupt_change_quantile=1.0,
)

REFERENCE = LightReference(
    median=30.0,
    standard_deviation=10.0,
    minimum=10.0,
    maximum=50.0,
    stuck_low=12.0,
    stuck_high=48.0,
    range_low=10.0,
    range_high=50.0,
    abrupt_change_threshold=100.0,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Light": [10.0, 20.0, 30.0, 40.0, 50.0, 45.0, 35.0, 25.0],
            "Temperature": np.arange(8, dtype=float),
            "Occupancy": [0, 0, 1, 1, 1, 1, 0, 0],
        }
    )


def test_fault_injection_changes_only_light_inside_episode() -> None:
    original = _frame()
    modified, truth = inject_light_fault(
        original,
        scenario="stuck_low",
        start=2,
        length=3,
        reference=REFERENCE,
    )

    assert original["Light"].tolist() == _frame()["Light"].tolist()
    assert modified.loc[2:4, "Light"].tolist() == [12.0, 12.0, 12.0]
    assert modified.drop(columns="Light").equals(original.drop(columns="Light"))
    assert truth.tolist() == [False, False, True, True, True, False, False, False]


def test_missing_detector_flags_current_row_immediately() -> None:
    frame, _ = inject_light_fault(
        _frame(),
        scenario="missing",
        start=3,
        length=2,
        reference=REFERENCE,
    )
    detected = detect_light_faults(
        frame["Light"], parameters=PARAMETERS, reference=REFERENCE
    )

    assert detected.loc[3, "missing_rule"]
    assert detected.loc[3, "detected_light_fault"]


def test_stuck_detector_uses_trailing_window_and_waits_for_evidence() -> None:
    light = pd.Series([10.0, 20.0, 30.0, 30.0, 30.0, 30.0])
    detected = detect_light_faults(
        light, parameters=PARAMETERS, reference=REFERENCE
    )

    assert not detected.loc[3, "stuck_rule"]
    assert detected.loc[4, "stuck_rule"]
    assert detected.loc[5, "stuck_rule"]


def test_out_of_range_detector_flags_implausibly_high_value() -> None:
    light = pd.Series([20.0, 30.0, 60.0])
    detected = detect_light_faults(
        light, parameters=PARAMETERS, reference=REFERENCE
    )

    assert detected.loc[2, "range_rule"]
    assert detected.loc[2, "detected_light_fault"]


def test_constant_normal_darkness_is_not_called_stuck() -> None:
    light = pd.Series([12.0, 12.0, 12.0, 12.0])
    detected = detect_light_faults(
        light, parameters=PARAMETERS, reference=REFERENCE
    )

    assert not detected["stuck_rule"].any()


def test_fault_metrics_include_false_alarms_and_detection_delay() -> None:
    truth = np.array([False, False, True, True, True, False])
    detected = np.array([True, False, False, True, True, False])
    metrics = fault_detection_metrics(truth, detected)

    assert metrics["detection_precision"] == pytest.approx(2 / 3)
    assert metrics["detection_recall"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["detection_delay_rows"] == 1


def test_routing_uses_fallback_only_where_mask_is_true() -> None:
    routed = route_probabilities(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.8, 0.7, 0.6]),
        np.array([False, True, False]),
    )

    assert routed.tolist() == pytest.approx([0.1, 0.7, 0.3])


def test_detector_is_prefix_invariant_and_does_not_use_future_rows() -> None:
    light = pd.Series([20.0, 30.0, 30.0, 30.0, 40.0, 50.0])
    full = detect_light_faults(
        light, parameters=PARAMETERS, reference=REFERENCE
    )
    prefix = detect_light_faults(
        light.iloc[:4], parameters=PARAMETERS, reference=REFERENCE
    )

    assert full.iloc[:4].equals(prefix)


def test_configured_parameter_grid_has_expected_cartesian_size() -> None:
    config = {
        "stuck_windows": [5, 10, 20],
        "stuck_tolerances_lux": [0.0, 1.0, 5.0],
        "range_quantile_pairs": [
            {"low": 0.0, "high": 1.0},
            {"low": 0.001, "high": 0.999},
        ],
        "abrupt_change_quantiles": [0.99, 1.0],
    }

    assert len(build_parameter_grid(config)) == 3 * 3 * 2 * 2


def test_linear_bias_endpoint_and_unchanged_other_columns() -> None:
    original = _frame()
    modified, truth = inject_light_fault(
        original,
        scenario="linear_bias_positive",
        start=2,
        length=4,
        reference=REFERENCE,
        linear_bias_final_std=1.0,
    )

    assert modified.loc[2, "Light"] == pytest.approx(original.loc[2, "Light"])
    assert modified.loc[5, "Light"] == pytest.approx(
        original.loc[5, "Light"] + REFERENCE.standard_deviation
    )
    assert modified.drop(columns="Light").equals(original.drop(columns="Light"))
    assert truth.sum() == 4
