from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensorbudget.modeling.schema import TARGET_COLUMN
from sensorbudget.robustness.fault_aware import (
    MISSING_INDICATOR,
    augment_training_frame,
    load_fault_aware_config,
    prepare_model_input,
)
from sensorbudget.robustness.fault_detection import LightReference

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


def _frame(rows: int = 14) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Temperature": np.linspace(20.0, 22.0, rows),
            "Light": np.linspace(10.0, 50.0, rows),
            "CO2": np.linspace(500.0, 900.0, rows),
            "Occupancy": np.resize([0, 1], rows),
        }
    )


def test_prepare_model_input_imputes_and_records_missing_light() -> None:
    frame = _frame(4)
    frame.loc[1, "Light"] = np.nan

    prepared = prepare_model_input(
        frame, light_median=30.0, include_missing_indicator=True
    )

    assert prepared.loc[1, "Light"] == pytest.approx(30.0)
    assert prepared[MISSING_INDICATOR].tolist() == [0, 1, 0, 0]
    assert not prepared.isna().any().any()


def test_prepare_model_input_omits_indicator_when_not_requested() -> None:
    prepared = prepare_model_input(
        _frame(4), light_median=30.0, include_missing_indicator=False
    )

    assert MISSING_INDICATOR not in prepared


def test_augmentation_preserves_clean_rows_and_adds_requested_fraction() -> None:
    original = _frame()
    scenarios = [
        "missing",
        "stuck_current",
        "stuck_low",
        "stuck_high",
        "out_of_range_high",
        "linear_bias_positive",
        "linear_bias_negative",
    ]

    augmented = augment_training_frame(
        original,
        scenarios=scenarios,
        reference=REFERENCE,
        linear_bias_final_std=1.0,
        out_of_range_std_multiplier=1.0,
        augmentation_ratio=0.5,
    )

    assert len(augmented) == len(original) + round(0.5 * len(original))
    pd.testing.assert_frame_equal(
        augmented.iloc[: len(original)].reset_index(drop=True), original
    )
    assert augmented.iloc[len(original) :]["Light"].isna().any()
    assert set(augmented[TARGET_COLUMN]) <= set(original[TARGET_COLUMN])


def test_config_loads_declared_representations() -> None:
    config = load_fault_aware_config("configs/fault_aware_training.json")

    assert config["representations"] == [
        "fault_aware",
        "fault_aware_missing_indicator",
    ]
