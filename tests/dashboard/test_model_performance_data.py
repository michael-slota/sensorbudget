"""Contract tests for the committed model-performance dashboard data."""

import json
from pathlib import Path


def load_payload() -> dict[str, object]:
    """Load the publishable aggregate without requiring ignored model files."""

    return json.loads(
        Path("site/data/model-performance.json").read_text(encoding="utf-8")
    )


def test_model_dashboard_identifies_both_candidate_stages() -> None:
    payload = load_payload()

    assert payload["metadata"]["threshold"] == 0.5
    assert "histogram gradient boosting" in payload["metadata"][
        "baseline_candidate"
    ].lower()
    assert "logistic regression" in payload["metadata"]["research_candidate"].lower()


def test_research_candidate_headline_f1_matches_model_card() -> None:
    payload = load_payload()
    rows = {row["split"]: row for row in payload["primary_confusion"]}

    assert round(rows["test_1"]["f1"], 3) == 0.971
    assert round(rows["test_2"]["f1"], 3) == 0.980
    assert len(payload["precision_recall_curves"]) == 4
