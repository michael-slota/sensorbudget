"""Contract tests for the committed sensor-selection dashboard data."""

import json
from pathlib import Path


def load_payload() -> dict[str, object]:
    """Load the publishable aggregate without ignored model artifacts."""

    return json.loads(
        Path("site/data/sensor-selection.json").read_text(encoding="utf-8")
    )


def test_all_sensor_combinations_are_published() -> None:
    payload = load_payload()

    assert payload["metadata"]["configuration_count"] == 15
    assert len(payload["validation_ranking"]) == 15
    assert len(payload["light_additions"]) == 7


def test_core_frontier_and_leading_validation_result_match_phase_four() -> None:
    payload = load_payload()
    leading = payload["validation_ranking"][0]
    always_frontier = {
        row["sensor_label"]
        for row in payload["frontier_frequency"]
        if row["frontier_scenarios"] == 5
    }

    assert leading["sensor_label"] == "Temperature + Light + CO2"
    assert round(leading["cv_f1_mean"], 3) == 0.780
    assert always_frontier == {
        "Light",
        "Humidity + Light",
        "Temperature + Light + CO2",
    }
