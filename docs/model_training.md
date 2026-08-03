# Baseline model training guide

## Purpose

The baseline modeling pipeline answers two initial questions:

1. Which classifier performs best when predicting later time periods?
2. How much performance is lost when the Light sensor is unavailable?

This document explains how the code works. Experimental results and their
interpretation are recorded separately in
[`reports/baseline_results.md`](../reports/baseline_results.md).

## Running the experiment

From the repository root, make the package importable:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Then run:

```powershell
python -m sensorbudget.modeling.train
```

If the package has been installed in editable mode, the equivalent command is:

```powershell
sensorbudget-train-baselines
```

The command executes `main()` in
[`src/sensorbudget/modeling/train.py`](../src/sensorbudget/modeling/train.py).
`main()` reads command-line arguments and passes them to `train_baselines()`,
which controls the experiment.

## End-to-end workflow

```text
Verify raw-file checksums
          ↓
Load and validate train, test_1, and test_2
          ↓
Define all-sensor and no-Light feature sets
          ↓
Evaluate five model types using chronological validation
          ↓
Select one model independently for each feature set
          ↓
Retrain both selected models on the complete training period
          ↓
Evaluate selected models on test_1 and test_2
          ↓
Save models, predictions, metrics, and metadata
```

## Code organization

| Module | Responsibility |
|---|---|
| `modeling/schema.py` | Target, feature sets, paths, seed, threshold, and CV defaults |
| `modeling/models.py` | Candidate estimator definitions |
| `modeling/evaluate.py` | Chronological CV, probability extraction, metrics, and selection |
| `modeling/artifacts.py` | Model, CSV, and JSON persistence |
| `modeling/train.py` | End-to-end orchestration and command-line interface |

The Plotly notebook
[`notebooks/02_baseline_models.ipynb`](../notebooks/02_baseline_models.ipynb)
reads generated artifacts and presents them. It does not implement or control
model training.

## Step 1: source verification

Training begins by verifying the raw files:

```python
validate_source_checksums(raw_dir, checksum_path)
```

The current source files must match the sizes and SHA-256 hashes committed in
`data/source_checksums.json`. Training stops if a file is missing, modified, or
corrupted.

The three supplied periods are then loaded and validated:

```python
frames = load_source_splits(raw_dir)
validate_source_splits(frames)
```

They remain separately accessible as:

```python
frames["train"]
frames["test_1"]
frames["test_2"]
```

Only `train` is used for model comparison and selection.

## Step 2: target and feature sets

The prediction target is:

```python
TARGET_COLUMN = "Occupancy"
```

Two feature configurations are evaluated:

```python
FEATURE_SETS = {
    "all_sensors": [
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
    ],
    "no_light": [
        "Temperature",
        "Humidity",
        "CO2",
        "HumidityRatio",
    ],
}
```

The following columns are excluded from model inputs:

- `source_row_id`: identifies a row but does not describe room conditions;
- `date`: intentionally excluded from the initial sensor-only baseline;
- `source_split`: reveals the source period and would cause leakage;
- `Occupancy`: the answer the model must predict.

The no-Light configuration tests whether environmental measurements can support
occupancy detection without relying on lighting behavior.

## Step 3: candidate estimators

`build_baseline_estimators()` creates five fixed candidates.

### Dummy prior

```python
DummyClassifier(strategy="prior")
```

The dummy model ignores all sensor readings. It provides a minimum benchmark
that learned models must exceed.

### Logistic regression

```python
Pipeline(
    [
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
            ),
        ),
    ]
)
```

Standardization puts differently scaled measurements—such as CO2 and
HumidityRatio—on comparable numerical scales. Balanced class weights increase
the influence of the less common occupied class.

### Decision tree

```python
DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=8,
    min_samples_leaf=20,
)
```

The tree learns nonlinear threshold rules. Depth and minimum leaf size are
restricted to reduce overfitting.

### Random forest

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    min_samples_leaf=5,
)
```

The forest combines 300 trees to produce a more stable probability estimate
than one decision tree.

### Histogram gradient boosting

```python
HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    class_weight="balanced",
)
```

Boosting builds trees sequentially, with later trees correcting mistakes made
by the current ensemble.

These are baseline configurations, not fully tuned hyperparameters.

## Step 4: chronological validation

The training period is sorted by timestamp and passed to:

```python
TimeSeriesSplit(n_splits=5)
```

This creates expanding chronological folds:

```text
Fold 1: early training rows       → next validation block
Fold 2: larger training history   → next validation block
Fold 3: still larger history      → next validation block
Fold 4: still larger history      → next validation block
Fold 5: most training history     → final validation block
```

Every validation observation occurs later than the observations used to fit its
model. Rows are never randomly shuffled.

This prevents adjacent minute-level observations from the same occupancy
episode being distributed randomly across development and validation data.

One validation block covers an all-unoccupied weekend. It is retained because
it is a real chronological operating condition. Occupied-class F1 is zero in
that fold, while PR-AUC and ROC-AUC are undefined because both target classes
are required.

## Step 5: model fitting

For every fold, the code evaluates:

```text
5 model types × 2 feature sets = 10 configurations
```

Across five folds:

```text
10 configurations × 5 folds = 50 validation fits
```

Before each fit, scikit-learn creates a fresh estimator:

```python
fitted = clone(estimator)
```

This prevents learned state from one fold carrying into another.

The estimator is then fitted using only the current fold's earlier rows:

```python
fitted.fit(
    development[feature_columns],
    development["Occupancy"],
)
```

Preprocessing inside the logistic-regression pipeline is also fitted only on
those development rows.

## Step 6: probabilities and threshold

Each fitted model produces an occupied-class probability:

```python
probability = fitted.predict_proba(features)[:, 1]
```

The baseline decision threshold is 0.5:

```python
predicted = (probability >= 0.5).astype(int)
```

Threshold 0.5 has not been optimized. Later threshold selection must use
training-period validation and explicit operational costs, not held-out tests.

## Step 7: metrics

The pipeline calculates:

| Metric | Purpose |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Balanced accuracy | Average recall across both target classes |
| Precision | Share of occupied predictions that are correct |
| Recall | Share of occupied observations detected |
| F1 | Harmonic balance of occupied precision and recall |
| Average precision / PR-AUC | Probability ranking for the minority class |
| ROC-AUC | Ranking across occupied and unoccupied observations |
| Brier score | Mean squared error of predicted probabilities |
| Confusion counts | True/false positives and negatives |

F1 is the primary model-selection metric because occupancy is the minority
class and both missed occupancy and false occupied predictions matter.

## Step 8: model selection

Fold results are aggregated by model and feature set:

```python
cv_summary = summarize_cross_validation(fold_metrics)
```

One model is selected independently for each feature set:

```python
selected = select_best_models(cv_summary)
```

Candidates are ranked by:

1. highest mean chronological F1;
2. highest mean average precision as a tie-breaker;
3. lowest mean Brier score as a second tie-breaker.

The current selected configurations are:

```text
all_sensors → histogram gradient boosting
no_light    → logistic regression
```

## Step 9: final fitting

Selection is now complete. Each winning estimator is recreated and trained on
all 8,143 rows in the supplied training period:

```python
fitted.fit(
    training[feature_columns],
    training["Occupancy"],
)
```

This adds two final fits:

```text
50 validation fits + 2 final fits = 52 total fits
```

## Step 10: held-out evaluation

Only the two selected configurations are evaluated on `test_1` and `test_2`.

The held-out periods do not influence:

- feature-set definition;
- candidate comparison;
- model selection;
- the current threshold.

The results provide an estimate of how each selected model behaves in distinct
held-out collection periods: Test 1 precedes training and Test 2 follows it.
The tests are reported separately because their
occupancy prevalence and sensor distributions differ.

## Step 11: generated artifacts

Training writes artifacts under `models/baseline/`:

| Artifact | Contents |
|---|---|
| `cv_fold_metrics.csv` | Every fold, model, feature set, and metric |
| `cv_predictions.csv` | Row-level chronological validation probabilities |
| `cv_summary.csv` | Mean and standard deviation across folds |
| `heldout_metrics.csv` | Final metrics for both selected models and tests |
| `heldout_predictions.csv` | Row-level probabilities and predictions |
| `metadata.json` | Settings, versions, selected models, and environment |
| `*.joblib` | Fitted estimator plus its feature and threshold contract |

These files are reproducible and ignored by Git.

Each saved model bundle contains:

```python
{
    "feature_set": ...,
    "model_name": ...,
    "feature_columns": ...,
    "threshold": ...,
    "estimator": ...,
}
```

The explicit feature list prevents prediction code from silently using columns
in the wrong order.

## Tests

Modeling tests under `tests/modeling/` verify:

- confusion-matrix counts and F1 calculations;
- behavior when a validation fold contains one target class;
- chronological ordering of validation folds;
- aggregation and F1-based model selection.

Run all tests with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q -p no:cacheprovider
```

## Current limitations

- Hyperparameters have not been systematically tuned.
- Calendar and lagged features are not included.
- The threshold remains at 0.5.
- Probability calibration has not been optimized.
- Results come from one room over a short collection period.
- Light may behave differently under other lighting policies or sensor faults.

These limitations are addressed in later sensor-ablation, robustness,
calibration, and deployment phases.
