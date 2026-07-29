from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from sensorbudget.modeling.evaluate import (
    classification_metrics,
    select_best_models,
    summarize_cross_validation,
    time_series_cross_validate,
)


def test_classification_metrics_returns_confusion_counts() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.8, 0.4]),
        threshold=0.5,
    )

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["f1"] == pytest.approx(0.5)


def test_single_class_fold_has_undefined_ranking_metrics() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 0]),
        np.array([0.1, 0.2, 0.3]),
    )

    assert np.isnan(metrics["average_precision"])
    assert np.isnan(metrics["roc_auc"])


def test_time_series_validation_preserves_fold_order() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="min"),
            "Temperature": np.linspace(20, 22, 12),
            "Occupancy": [0, 0, 1, 1] * 3,
        }
    )
    metrics = time_series_cross_validate(
        frame,
        {"dummy": DummyClassifier(strategy="prior")},
        {"temperature": ["Temperature"]},
        n_splits=3,
    )

    assert len(metrics) == 3
    assert metrics["validation_start"].is_monotonic_increasing
    assert (metrics["validation_start"] > frame["date"].min()).all()


def test_cv_summary_and_selection_use_mean_f1() -> None:
    folds = pd.DataFrame(
        {
            "feature_set": ["all", "all", "all", "all"],
            "model": ["a", "a", "b", "b"],
            "accuracy": [0.8, 0.8, 0.9, 0.9],
            "balanced_accuracy": [0.7, 0.7, 0.8, 0.8],
            "precision": [0.6, 0.6, 0.7, 0.7],
            "recall": [0.6, 0.6, 0.9, 0.9],
            "f1": [0.6, 0.6, 0.8, 0.8],
            "average_precision": [0.7, 0.7, 0.9, 0.9],
            "roc_auc": [0.7, 0.7, 0.9, 0.9],
            "brier_score": [0.2, 0.2, 0.1, 0.1],
        }
    )

    summary = summarize_cross_validation(folds)
    selected = select_best_models(summary)

    assert selected == {"all": "b"}

