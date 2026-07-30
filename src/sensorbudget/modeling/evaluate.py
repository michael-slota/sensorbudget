"""Time-aware model validation and held-out evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from sensorbudget.modeling.schema import (
    DEFAULT_CV_SPLITS,
    DEFAULT_THRESHOLD,
    TARGET_COLUMN,
)


def positive_class_probability(
    estimator: BaseEstimator,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return probability estimates for the occupied class."""

    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"{type(estimator).__name__} does not implement predict_proba()."
        )
    probabilities = estimator.predict_proba(features)
    return np.asarray(probabilities)[:, 1]


def classification_metrics(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | int]:
    """Calculate threshold and probability metrics for binary occupancy."""

    actual = np.asarray(target, dtype=int)
    score = np.asarray(probability, dtype=float)
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    has_both_classes = np.unique(actual).size == 2
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(actual, predicted))
            if has_both_classes
            else float("nan")
        ),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "average_precision": (
            float(average_precision_score(actual, score))
            if has_both_classes
            else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(actual, score))
            if has_both_classes
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(actual, score)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(threshold),
    }


def time_series_cross_validate(
    training_frame: pd.DataFrame,
    estimators: Mapping[str, BaseEstimator],
    feature_sets: Mapping[str, list[str]],
    *,
    n_splits: int = DEFAULT_CV_SPLITS,
    threshold: float = DEFAULT_THRESHOLD,
    return_predictions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model and feature set on expanding chronological folds."""

    ordered = training_frame.sort_values("date").reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(ordered),
        start=1,
    ):
        development = ordered.iloc[train_indices]
        validation = ordered.iloc[validation_indices]

        for feature_set_name, feature_columns in feature_sets.items():
            for model_name, estimator in estimators.items():
                fitted = clone(estimator)
                # Limit native worker pools for deterministic behavior in
                # restricted environments and to avoid nested parallelism.
                with threadpool_limits(limits=1):
                    fitted.fit(
                        development[feature_columns],
                        development[TARGET_COLUMN],
                    )
                probability = positive_class_probability(
                    fitted,
                    validation[feature_columns],
                )
                metrics = classification_metrics(
                    validation[TARGET_COLUMN],
                    probability,
                    threshold=threshold,
                )
                rows.append(
                    {
                        "fold": fold,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "train_rows": int(len(development)),
                        "validation_rows": int(len(validation)),
                        "train_occupied_rate": float(
                            development[TARGET_COLUMN].mean()
                        ),
                        "validation_occupied_rate": float(
                            validation[TARGET_COLUMN].mean()
                        ),
                        "validation_start": validation["date"].min(),
                        "validation_end": validation["date"].max(),
                        **metrics,
                    }
                )

                if return_predictions:
                    predictions = validation[
                        ["source_row_id", "date", "source_split", TARGET_COLUMN]
                    ].copy()
                    predictions.insert(0, "fold", fold)
                    predictions.insert(1, "feature_set", feature_set_name)
                    predictions.insert(2, "model", model_name)
                    predictions["probability_occupied"] = probability
                    predictions["predicted_occupancy"] = (
                        probability >= threshold
                    ).astype(int)
                    prediction_frames.append(predictions)

    metrics_frame = pd.DataFrame(rows)
    if not return_predictions:
        return metrics_frame
    return metrics_frame, pd.concat(prediction_frames, ignore_index=True)


def summarize_cross_validation(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics for model selection."""

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "brier_score",
    ]
    summary = (
        fold_metrics.groupby(["feature_set", "model"], as_index=False)[
            metric_columns
        ]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(column).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def select_best_models(cv_summary: pd.DataFrame) -> dict[str, str]:
    """Select the highest mean-F1 model independently for each feature set."""

    selections: dict[str, str] = {}
    for feature_set, candidates in cv_summary.groupby("feature_set"):
        ranked = candidates.sort_values(
            ["f1_mean", "average_precision_mean", "brier_score_mean"],
            ascending=[False, False, True],
            na_position="last",
        )
        selections[str(feature_set)] = str(ranked.iloc[0]["model"])
    return selections


def evaluate_fitted_model(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate one fitted model and return metrics plus row predictions."""

    probability = positive_class_probability(
        estimator,
        frame[feature_columns],
    )
    metrics = classification_metrics(
        frame[TARGET_COLUMN],
        probability,
        threshold=threshold,
    )
    predictions = frame[
        ["source_row_id", "date", "source_split", TARGET_COLUMN]
    ].copy()
    predictions["probability_occupied"] = probability
    predictions["predicted_occupancy"] = (
        probability >= threshold
    ).astype(int)
    return metrics, predictions
