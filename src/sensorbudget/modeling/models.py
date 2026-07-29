"""Baseline estimator definitions."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sensorbudget.modeling.schema import DEFAULT_RANDOM_SEED


def build_baseline_estimators(
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, BaseEstimator]:
    """Return fresh, fixed-configuration baseline classifiers."""

    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=5,
            max_features="sqrt",
            # Single-process fitting works in restricted Windows environments
            # and is fast enough for this small dataset.
            n_jobs=1,
            random_state=random_seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=random_seed,
        ),
    }
