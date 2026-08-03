"""Contract tests for the committed decision-and-explainability data."""

import json
from pathlib import Path


def load_payload() -> dict[str, object]:
    return json.loads(
        Path("site/data/decision-explainability.json").read_text(encoding="utf-8")
    )


def test_equal_cost_threshold_and_heldout_instability_are_preserved() -> None:
    payload = load_payload()
    equal = next(
        row for row in payload["selected_thresholds"] if row["scenario"] == "equal_cost"
    )
    test_two = {
        row["threshold_source"]: row
        for row in payload["heldout_equal_cost"]
        if row["split"] == "test_2"
    }
    assert equal["threshold"] == 0.86
    assert test_two["validation_selected"]["cost_per_1000_rows"] > test_two[
        "default_0.5"
    ]["cost_per_1000_rows"]


def test_global_explanation_identifies_light_as_dominant() -> None:
    payload = load_payload()
    strongest = max(
        payload["global_coefficients"],
        key=lambda row: abs(row["standardized_coefficient"]),
    )
    assert strongest["feature"] == "Light"
    assert round(strongest["standardized_coefficient"], 2) == 4.50
