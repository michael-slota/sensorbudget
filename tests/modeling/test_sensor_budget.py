from __future__ import annotations

from pathlib import Path

import pytest

from sensorbudget.modeling.sensor_budget import (
    build_sensor_scenarios,
    load_sensor_budget_config,
)

CONFIG_PATH = Path("configs/sensor_budget.json")


def test_sensor_budget_config_covers_all_physical_combinations() -> None:
    config = load_sensor_budget_config(CONFIG_PATH)
    feature_sets, scenarios = build_sensor_scenarios(config)

    assert len(feature_sets) == 15
    assert len(scenarios) == 15
    assert scenarios["physical_sensor_count"].value_counts().to_dict() == {
        1: 4,
        2: 6,
        3: 4,
        4: 1,
    }


def test_humidity_ratio_is_derived_only_when_inputs_are_available() -> None:
    config = load_sensor_budget_config(CONFIG_PATH)
    feature_sets, scenarios = build_sensor_scenarios(config)

    assert feature_sets["temperature__humidity"] == [
        "Temperature",
        "Humidity",
        "HumidityRatio",
    ]
    assert "HumidityRatio" not in feature_sets["temperature"]
    all_sensors = scenarios.loc[
        scenarios["feature_set"]
        == "temperature__humidity__light__co2"
    ].iloc[0]
    assert all_sensors["relative_cost"] == pytest.approx(6.5)
