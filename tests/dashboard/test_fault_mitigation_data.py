"""Contract tests for the committed fault-mitigation dashboard data."""

import json
from pathlib import Path


def load_payload() -> dict[str, object]:
    return json.loads(
        Path("site/data/fault-mitigation.json").read_text(encoding="utf-8")
    )


def test_fallback_selection_and_missing_detection_match_phase_five() -> None:
    payload = load_payload()
    selected = [
        row for row in payload["fallback_candidates"] if row["selected_fallback"]
    ]
    missing = [
        row for row in payload["detector_quality"] if row["scenario"] == "missing"
    ]
    assert len(selected) == 1
    assert selected[0]["feature_label"] == "Temperature + CO2"
    assert round(selected[0]["cv_f1_mean"], 3) == 0.679
    assert {round(row["detection_recall"], 3) for row in missing} == {1.0}


def test_strategy_aggregate_preserves_test_two_primary_advantage() -> None:
    payload = load_payload()
    test_two = {
        row["strategy"]: row["mean_f1"]
        for row in payload["strategy_average"]
        if row["split"] == "test_2"
    }
    assert test_two["primary_only"] > test_two["detector_routing"]
    assert round(test_two["detector_routing"], 3) == 0.974
