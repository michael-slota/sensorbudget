"""Contract tests for the committed robustness dashboard data."""

import json
from pathlib import Path


def load_payload() -> dict[str, object]:
    """Load the publishable aggregate without ignored model artifacts."""

    return json.loads(Path("site/data/robustness.json").read_text(encoding="utf-8"))


def test_robustness_dashboard_covers_three_frontier_configurations() -> None:
    payload = load_payload()
    configurations = {row["feature_label"] for row in payload["overview"]}

    assert payload["metadata"]["configuration_count"] == 3
    assert configurations == {
        "Light",
        "Humidity + Light",
        "Temperature + Light + CO2",
    }


def test_critical_light_failure_is_visible() -> None:
    payload = load_payload()
    critical = [
        row
        for row in payload["overview"]
        if row["display_scenario"]
        in {"Occupied but dark", "Complete Light loss"}
    ]

    assert critical
    assert max(row["f1_mean"] for row in critical) < 0.01
