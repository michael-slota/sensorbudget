from __future__ import annotations

from pathlib import Path

import pandas as pd

from sensorbudget.modeling.cost_sensitivity import (
    evaluate_cost_scenarios,
    load_cost_scenarios,
    mark_pareto_frontier,
)

SCENARIO_PATH = Path("configs/cost_scenarios.json")


def test_pareto_frontier_excludes_costlier_weaker_point() -> None:
    frame = pd.DataFrame(
        {
            "scenario_cost": [0.5, 1.5, 5.5, 6.5],
            "cv_f1_mean": [0.75, 0.76, 0.78, 0.77],
        }
    )

    membership = mark_pareto_frontier(frame)

    assert membership.tolist() == [True, True, True, False]


def test_pareto_frontier_keeps_only_best_point_at_equal_cost() -> None:
    frame = pd.DataFrame(
        {
            "scenario_cost": [1.0, 1.0, 2.0],
            "cv_f1_mean": [0.60, 0.70, 0.65],
        }
    )

    membership = mark_pareto_frontier(frame)

    assert membership.tolist() == [False, True, False]


def test_cost_scenarios_recalculate_additive_sensor_costs() -> None:
    config = load_cost_scenarios(SCENARIO_PATH)
    selected = pd.DataFrame(
        {
            "feature_set": ["light", "temperature__light__co2"],
            "physical_sensors": ["light", "temperature, light, co2"],
            "selected_model": ["a", "b"],
            "cv_f1_mean": [0.75, 0.78],
        }
    )

    results = evaluate_cost_scenarios(selected, config)
    current = results.loc[
        results["scenario"] == "current_assumptions"
    ].set_index("feature_set")

    assert current.loc["light", "scenario_cost"] == 0.5
    assert (
        current.loc["temperature__light__co2", "scenario_cost"]
        == 5.5
    )
    assert results["scenario"].nunique() == 5
