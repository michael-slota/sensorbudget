from __future__ import annotations

import numpy as np
import pandas as pd

from sensorbudget.robustness.evaluate import (
    apply_gaussian_noise,
    apply_gradual_drift,
    apply_light_policy_failure,
    apply_random_missingness,
    stable_random_seed,
)


def test_stable_random_seed_is_reproducible_and_context_specific() -> None:
    first = stable_random_seed(42, "light", "test_1", 0.1, 1)
    repeated = stable_random_seed(42, "light", "test_1", 0.1, 1)
    different = stable_random_seed(42, "light", "test_1", 0.2, 1)

    assert first == repeated
    assert first != different


def test_random_missingness_uses_provided_medians() -> None:
    frame = pd.DataFrame(
        {
            "Light": np.arange(20, dtype=float),
            "CO2": np.arange(100, 120, dtype=float),
        }
    )

    modified, affected = apply_random_missingness(
        frame,
        ["Light", "CO2"],
        rate=0.5,
        medians={"Light": 7.0, "CO2": 107.0},
        rng=np.random.default_rng(42),
    )

    assert affected > 0
    assert not modified.isna().any().any()
    assert (modified["Light"] == 7.0).any()
    assert (modified["CO2"] == 107.0).any()


def test_gaussian_noise_is_bounded_by_training_range() -> None:
    frame = pd.DataFrame({"Light": np.full(100, 50.0)})

    modified = apply_gaussian_noise(
        frame,
        ["Light"],
        std_fraction=2.0,
        standard_deviations={"Light": 20.0},
        observed_minimums={"Light": 0.0},
        observed_maximums={"Light": 100.0},
        rng=np.random.default_rng(42),
    )

    assert modified["Light"].between(0.0, 100.0).all()
    assert not modified["Light"].equals(frame["Light"])


def test_gradual_drift_starts_at_zero_and_reaches_requested_offset() -> None:
    frame = pd.DataFrame({"CO2": np.full(5, 500.0)})

    modified = apply_gradual_drift(
        frame,
        "CO2",
        final_std_fraction=0.5,
        training_standard_deviation=100.0,
    )

    assert modified["CO2"].iloc[0] == 500.0
    assert modified["CO2"].iloc[-1] == 550.0


def test_light_policy_failures_modify_only_target_class() -> None:
    frame = pd.DataFrame(
        {
            "Light": [0.0, 10.0, 400.0, 450.0],
            "Occupancy": [0, 0, 1, 1],
        }
    )

    unoccupied_lit, lit_count = apply_light_policy_failure(
        frame,
        mode="unoccupied_lit",
        occupied_light_reference=425.0,
        unoccupied_light_reference=5.0,
    )
    occupied_dark, dark_count = apply_light_policy_failure(
        frame,
        mode="occupied_dark",
        occupied_light_reference=425.0,
        unoccupied_light_reference=5.0,
    )

    assert lit_count == 2
    assert dark_count == 2
    assert unoccupied_lit.loc[:1, "Light"].eq(425.0).all()
    assert unoccupied_lit.loc[2:, "Light"].equals(frame.loc[2:, "Light"])
    assert occupied_dark.loc[2:, "Light"].eq(5.0).all()
    assert occupied_dark.loc[:1, "Light"].equals(frame.loc[:1, "Light"])
