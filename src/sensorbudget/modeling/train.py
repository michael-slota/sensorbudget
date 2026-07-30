"""Train time-aware baseline models and evaluate selected finalists."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from sensorbudget.data.load import load_source_splits
from sensorbudget.data.provenance import validate_source_checksums
from sensorbudget.data.schema import (
    DEFAULT_CHECKSUM_PATH,
    DEFAULT_RAW_DIR,
)
from sensorbudget.data.validate import validate_source_splits
from sensorbudget.modeling.artifacts import (
    save_model_bundle,
    write_json,
    write_table,
)
from sensorbudget.modeling.evaluate import (
    evaluate_fitted_model,
    select_best_models,
    summarize_cross_validation,
    time_series_cross_validate,
)
from sensorbudget.modeling.models import build_baseline_estimators
from sensorbudget.modeling.schema import (
    DEFAULT_CV_SPLITS,
    DEFAULT_MODEL_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_THRESHOLD,
    FEATURE_SETS,
    TARGET_COLUMN,
)


def _json_metrics(metrics: dict[str, float | int]) -> dict[str, Any]:
    return {
        key: (
            None
            if isinstance(value, float) and np.isnan(value)
            else value
        )
        for key, value in metrics.items()
    }


def train_baselines(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    output_dir: Path | str = DEFAULT_MODEL_DIR,
    n_splits: int = DEFAULT_CV_SPLITS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Run validation, CV model selection, final fitting, and held-out tests."""

    validate_source_checksums(raw_dir, checksum_path)
    frames = load_source_splits(raw_dir)
    validate_source_splits(frames)

    training = frames["train"].sort_values("date").reset_index(drop=True)
    estimators = build_baseline_estimators(random_seed)
    fold_metrics, cv_predictions = time_series_cross_validate(
        training,
        estimators,
        FEATURE_SETS,
        n_splits=n_splits,
        threshold=threshold,
        return_predictions=True,
    )
    cv_summary = summarize_cross_validation(fold_metrics)
    selected = select_best_models(cv_summary)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "cv_fold_metrics.csv", fold_metrics)
    write_table(output / "cv_predictions.csv", cv_predictions)
    write_table(output / "cv_summary.csv", cv_summary)

    heldout_rows = []
    prediction_frames = []
    model_records: dict[str, Any] = {}

    for feature_set, model_name in selected.items():
        feature_columns = FEATURE_SETS[feature_set]
        fitted = clone(estimators[model_name])
        with threadpool_limits(limits=1):
            fitted.fit(training[feature_columns], training[TARGET_COLUMN])
        model_path = save_model_bundle(
            output,
            feature_set,
            model_name,
            fitted,
            feature_columns,
            threshold,
        )

        model_records[feature_set] = {
            "model": model_name,
            "features": feature_columns,
            "artifact": model_path.name,
        }
        for split in ("test_1", "test_2"):
            metrics, predictions = evaluate_fitted_model(
                fitted,
                frames[split],
                feature_columns,
                threshold=threshold,
            )
            heldout_rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "split": split,
                    **metrics,
                }
            )
            predictions.insert(0, "feature_set", feature_set)
            predictions.insert(1, "model", model_name)
            prediction_frames.append(predictions)

    heldout_metrics = pd.DataFrame(heldout_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_table(output / "heldout_metrics.csv", heldout_metrics)
    write_table(output / "heldout_predictions.csv", predictions)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "cv_splits": n_splits,
        "threshold": threshold,
        "selection_metric": "mean chronological CV F1",
        "training_rows": int(len(training)),
        "training_start": training["date"].min().isoformat(),
        "training_end": training["date"].max().isoformat(),
        "selected_models": model_records,
        "heldout_metrics": [
            {
                key: value
                for key, value in {
                    **row,
                    **_json_metrics(
                        {
                            metric: row[metric]
                            for metric in heldout_metrics.columns
                            if metric
                            not in {"feature_set", "model", "split"}
                        }
                    ),
                }.items()
            }
            for row in heldout_metrics.to_dict(orient="records")
        ],
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json(output / "metadata.json", metadata)
    return {
        "fold_metrics": fold_metrics,
        "cv_predictions": cv_predictions,
        "cv_summary": cv_summary,
        "selected_models": selected,
        "heldout_metrics": heldout_metrics,
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run baseline training from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--checksums",
        type=Path,
        default=DEFAULT_CHECKSUM_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    result = train_baselines(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        output_dir=args.output_dir,
        n_splits=args.cv_splits,
        random_seed=args.seed,
        threshold=args.threshold,
    )
    print("Selected models:")
    print(json.dumps(result["selected_models"], indent=2))
    print("\nHeld-out metrics:")
    print(result["heldout_metrics"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
