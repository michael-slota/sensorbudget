"""Contract tests for the committed static EDA dashboard data."""

import json
from pathlib import Path


def load_payload() -> dict[str, object]:
    """Load the versioned aggregate without requiring ignored raw data."""

    return json.loads(Path("site/data/eda.json").read_text(encoding="utf-8"))


def test_eda_payload_matches_audited_source_totals() -> None:
    payload = load_payload()

    assert payload["metadata"]["total_rows"] == 20_560
    assert [row["rows"] for row in payload["split_summary"]] == [8_143, 2_665, 9_752]
    assert [row["occupied_rows"] for row in payload["split_summary"]] == [
        1_729,
        972,
        2_049,
    ]


def test_light_proxy_and_drift_evidence_are_preserved() -> None:
    payload = load_payload()
    light_rows = {row["split"]: row for row in payload["light_exceptions"]}
    correlations = {
        row["sensor"]: row["correlation"]
        for row in payload["target_correlations"]
    }

    assert light_rows["train"]["occupied_while_dark"] == 0
    assert light_rows["test_1"]["occupied_while_dark"] == 0
    assert light_rows["test_2"]["occupied_while_dark"] == 1
    assert correlations["Light"] > 0.9
