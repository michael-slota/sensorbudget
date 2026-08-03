# Baseline occupancy model results

## Executive summary

- Five fixed classifiers were compared with expanding chronological
  validation; histogram gradient boosting led the initial all-sensor study.
- Its held-out F1 was `0.930` on Test 1 and `0.883` on Test 2, showing that
  strong ranking metrics did not translate into identical period performance.
- This is the initial baseline, not the later three-sensor logistic research
  candidate used for robustness and explainability.

> **Project status:** This report records the Phase 3 baseline comparison. The
> later sensor-budget study selected a separate Temperature + Light + CO2
> logistic research candidate for Phases 5–6. See the
> [sensor-budget report](sensor_budget_results.md) and consolidated
> [model card](model_card.md).

**Traceability:** Settings are versioned in
[`configs/baseline.yaml`](../configs/baseline.yaml), the presentation is in
[`notebooks/02_baseline_models.ipynb`](../notebooks/02_baseline_models.ipynb),
and source identity is fixed by
[`data/source_checksums.json`](../data/source_checksums.json). The Git commit
containing this report identifies the corresponding code revision.

## Experiment design

Five fixed baseline classifiers were compared:

- prior-probability dummy classifier;
- class-balanced logistic regression with standardized inputs;
- class-balanced decision tree;
- class-balanced random forest;
- class-balanced histogram gradient boosting.

Each model was evaluated with:

- `all_sensors`: Temperature, Humidity, Light, CO2, and HumidityRatio;
- `no_light`: the same set with Light removed.

Model selection used five expanding chronological folds within the supplied
training period. The decision threshold remained at 0.5. The two supplied test
periods were evaluated only after selecting one model per feature set.

One validation fold covers a weekend period containing no occupied rows. Its F1
is correctly zero because there are no positive events to recover. PR-AUC and
ROC-AUC are undefined for that fold and are excluded from their respective
means. The weekend is retained because it represents a genuine chronological
deployment condition.

## Chronological validation

| Feature set | Model | Mean F1 | F1 SD | Mean precision | Mean recall |
|---|---|---:|---:|---:|---:|
| all sensors | Histogram gradient boosting | **0.767** | 0.432 | 0.741 | 0.799 |
| all sensors | Random forest | 0.726 | 0.429 | 0.687 | 0.798 |
| all sensors | Logistic regression | 0.716 | 0.425 | 0.766 | 0.697 |
| all sensors | Decision tree | 0.699 | 0.425 | 0.671 | 0.776 |
| all sensors | Dummy prior | 0.000 | 0.000 | 0.000 | 0.000 |
| no Light | Logistic regression | **0.617** | 0.368 | 0.565 | 0.740 |
| no Light | Random forest | 0.609 | 0.348 | 0.548 | 0.745 |
| no Light | Histogram gradient boosting | 0.595 | 0.344 | 0.533 | 0.745 |
| no Light | Decision tree | 0.588 | 0.347 | 0.525 | 0.733 |
| no Light | Dummy prior | 0.000 | 0.000 | 0.000 | 0.000 |

Selected finalists:

- all sensors: histogram gradient boosting;
- no Light: logistic regression.

The large fold-to-fold standard deviations reflect changing occupancy
prevalence and the all-unoccupied weekend fold. Fold-level results should be
inspected rather than interpreting the mean alone.

## Held-out results

| Feature set | Test | F1 | Precision | Recall | Balanced accuracy | PR-AUC | ROC-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all sensors | test 1 | **0.930** | 0.940 | 0.920 | 0.943 | 0.976 | 0.990 | 0.040 |
| all sensors | test 2 | **0.883** | 0.830 | 0.943 | 0.946 | 0.962 | 0.991 | 0.039 |
| no Light | test 1 | 0.832 | 0.777 | 0.894 | 0.873 | 0.913 | 0.944 | 0.104 |
| no Light | test 2 | 0.546 | 0.435 | 0.731 | 0.739 | 0.360 | 0.763 | 0.222 |

Confusion counts:

| Feature set | Test | False positive | False negative |
|---|---|---:|---:|
| all sensors | test 1 | 57 | 78 |
| all sensors | test 2 | 397 | 116 |
| no Light | test 1 | 249 | 103 |
| no Light | test 2 | 1,945 | 551 |

## Interpretation

The all-sensor model performs strongly in both held-out periods. Its probability
ranking is particularly stable, with ROC-AUC near 0.99 in both tests.

Removing Light causes a modest reduction in `test_1` but a substantial loss in
`test_2`. The no-Light model produces 1,945 false occupied predictions in
`test_2`, resulting in precision of only 0.435. This is consistent with the EDA:
environmental conditions—especially Humidity and HumidityRatio—shift strongly
in `test_2`.

These results support two conclusions:

1. Light is exceptionally valuable for this particular room and collection
   period.
2. A model without Light is possible, but its default 0.5 threshold and
   cross-period robustness are not yet adequate for deployment.

## Limitations and next decisions

- Hyperparameters are fixed baselines, not tuned configurations.
- The 0.5 decision threshold has not been optimized for operational costs.
- Calendar features and temporal lag features are intentionally excluded.
- Test results describe one room over a short period and do not establish
  generalization to other buildings.
- The Light relationship may change under daylight, automated lighting, or
  sensor failure.

The subsequent sensor-subset ablations and cost-scenario analysis are reported
in `reports/sensor_budget_results.md`. Robustness, threshold, and calibration
decisions should use training-period validation—not the held-out tests
reported here.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m sensorbudget.modeling.train
```

Generated artifacts are written to `models/baseline/` and include fold metrics,
row-level chronological validation probabilities, summary metrics, held-out
predictions, fitted model bundles, and environment metadata.
