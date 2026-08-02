from __future__ import annotations

import pandas as pd
import pytest

from sensorbudget.robustness.fallback import (
    select_fallback_candidate,
    validate_fallback_features,
)


def _selected_models() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_set": ["temperature", "co2", "temperature__co2"],
            "selected_model": ["tree", "logistic", "logistic"],
            "cv_f1_mean": [0.41, 0.67, 0.68],
        }
    )


def test_fallback_selection_uses_highest_training_cv_f1() -> None:
    selected = select_fallback_candidate(
        _selected_models(),
        ["temperature", "co2", "temperature__co2"],
    )

    assert selected["feature_set"] == "temperature__co2"
    assert selected["cv_f1_mean"] == pytest.approx(0.68)


def test_fallback_selection_rejects_unknown_candidate() -> None:
    with pytest.raises(ValueError, match="Unknown fallback candidates"):
        select_fallback_candidate(_selected_models(), ["humidity"])


def test_fallback_features_must_exclude_light() -> None:
    with pytest.raises(ValueError, match="must not use Light"):
        validate_fallback_features(
            {"Temperature", "Light", "CO2"},
            {"Light", "CO2"},
        )


def test_fallback_features_must_exist_in_primary_system() -> None:
    with pytest.raises(ValueError, match="already be available"):
        validate_fallback_features(
            {"Temperature", "Light", "CO2"},
            {"Temperature", "Humidity"},
        )


def test_valid_existing_non_light_features_are_accepted() -> None:
    validate_fallback_features(
        {"Temperature", "Light", "CO2"},
        {"Temperature", "CO2"},
    )
