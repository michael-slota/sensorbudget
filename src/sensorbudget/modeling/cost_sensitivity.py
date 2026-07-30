"""Recalculate sensor-budget frontiers under alternative cost scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sensorbudget.modeling.artifacts import write_json, write_table

DEFAULT_SCENARIO_PATH = Path("configs/cost_scenarios.json")
DEFAULT_SELECTED_MODELS_PATH = Path(
    "models/sensor_budget/selected_models.csv"
)
DEFAULT_OUTPUT_DIR = Path("models/sensor_budget")
PHYSICAL_SENSORS = frozenset(
    {"temperature", "humidity", "light", "co2"}
)


def load_cost_scenarios(
    path: Path | str = DEFAULT_SCENARIO_PATH,
) -> dict[str, Any]:
    """Load and validate named physical-sensor cost scenarios."""

    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    scenarios = config.get("scenarios", {})
    if not scenarios:
        raise ValueError("Cost config requires at least one scenario.")

    for scenario_name, scenario in scenarios.items():
        sensor_names = set(scenario) - {"label"}
        if sensor_names != PHYSICAL_SENSORS:
            raise ValueError(
                f"Scenario {scenario_name!r} must define exactly "
                f"{sorted(PHYSICAL_SENSORS)}."
            )
        if not str(scenario.get("label", "")).strip():
            raise ValueError(
                f"Scenario {scenario_name!r} requires a display label."
            )
        for sensor_name in PHYSICAL_SENSORS:
            if float(scenario[sensor_name]) <= 0:
                raise ValueError(
                    f"Scenario {scenario_name!r} has a non-positive "
                    f"cost for {sensor_name!r}."
                )
    return config


def mark_pareto_frontier(
    frame: pd.DataFrame,
    *,
    cost_column: str = "scenario_cost",
    score_column: str = "cv_f1_mean",
) -> pd.Series:
    """Mark rows not dominated by an equal-or-cheaper, equal-or-better row."""

    ordered = frame.sort_values(
        [cost_column, score_column],
        ascending=[True, False],
    )
    frontier_indices = []
    best_cheaper_score = -np.inf

    for _, cost_group in ordered.groupby(cost_column, sort=True):
        group_best = float(cost_group[score_column].max())
        if group_best > best_cheaper_score:
            frontier_indices.extend(
                cost_group.index[
                    np.isclose(cost_group[score_column], group_best)
                ].tolist()
            )
            best_cheaper_score = group_best

    return pd.Series(
        frame.index.isin(frontier_indices),
        index=frame.index,
        dtype=bool,
    )


def evaluate_cost_scenarios(
    selected_models: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply every cost scenario to fixed validation-selected configurations."""

    required_columns = {
        "feature_set",
        "physical_sensors",
        "selected_model",
        "cv_f1_mean",
    }
    missing = required_columns - set(selected_models.columns)
    if missing:
        raise ValueError(
            f"Selected-model table is missing columns: {sorted(missing)}."
        )

    result_frames = []
    for scenario_name, scenario in config["scenarios"].items():
        result = selected_models.copy()
        result.insert(0, "scenario", scenario_name)
        result.insert(1, "scenario_label", scenario["label"])
        result["scenario_cost"] = result["physical_sensors"].apply(
            lambda value: float(
                sum(
                    float(scenario[sensor])
                    for sensor in value.split(", ")
                )
            )
        )
        result["is_pareto"] = mark_pareto_frontier(result)
        result_frames.append(result)

    return pd.concat(result_frames, ignore_index=True)


def run_cost_sensitivity(
    *,
    selected_models_path: Path | str = DEFAULT_SELECTED_MODELS_PATH,
    scenario_path: Path | str = DEFAULT_SCENARIO_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Recalculate and save Pareto membership for all cost scenarios."""

    selected_source = Path(selected_models_path)
    scenario_source = Path(scenario_path)
    selected_models = pd.read_csv(selected_source)
    config = load_cost_scenarios(scenario_source)
    results = evaluate_cost_scenarios(selected_models, config)

    frontier = results.loc[
        results["is_pareto"],
        [
            "scenario",
            "scenario_label",
            "feature_set",
            "physical_sensors",
            "selected_model",
            "cv_f1_mean",
            "scenario_cost",
        ],
    ].sort_values(["scenario", "scenario_cost", "cv_f1_mean"])

    frequency = (
        results.groupby(
            ["feature_set", "physical_sensors"],
            as_index=False,
        )
        .agg(
            frontier_scenarios=("is_pareto", "sum"),
            total_scenarios=("scenario", "nunique"),
            cv_f1_mean=("cv_f1_mean", "first"),
        )
        .sort_values(
            ["frontier_scenarios", "cv_f1_mean"],
            ascending=[False, False],
        )
    )
    frequency["frontier_share"] = (
        frequency["frontier_scenarios"]
        / frequency["total_scenarios"]
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "cost_sensitivity.csv", results)
    write_table(output / "cost_sensitivity_frontiers.csv", frontier)
    write_table(output / "cost_sensitivity_frequency.csv", frequency)

    metadata = {
        "scenario_config_path": str(scenario_source),
        "scenario_config_sha256": hashlib.sha256(
            scenario_source.read_bytes()
        ).hexdigest(),
        "selected_models_path": str(selected_source),
        "selected_models_sha256": hashlib.sha256(
            selected_source.read_bytes()
        ).hexdigest(),
        "cost_unit": config["cost_unit"],
        "note": config["note"],
        "scenario_count": len(config["scenarios"]),
        "performance_metric": "mean chronological validation F1",
        "models_retrained": False,
    }
    write_json(output / "cost_sensitivity_metadata.json", metadata)
    return {
        "results": results,
        "frontier": frontier,
        "frequency": frequency,
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run cost-scenario sensitivity analysis from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-models",
        type=Path,
        default=DEFAULT_SELECTED_MODELS_PATH,
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    result = run_cost_sensitivity(
        selected_models_path=args.selected_models,
        scenario_path=args.scenarios,
        output_dir=args.output_dir,
    )
    print("Pareto frontiers by cost scenario:")
    print(result["frontier"].to_string(index=False))
    print("\nFrontier membership frequency:")
    print(result["frequency"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
