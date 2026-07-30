# Sensor-budget experiment

## Purpose

This experiment asks how occupancy-classification performance changes as
physical sensors are added or removed. It extends the baseline evaluation
without changing its leakage controls:

1. Preserve the supplied chronological training and test periods.
2. Compare five fixed model families within five expanding training folds.
3. Select one model independently for each sensor combination using mean
   validation F1.
4. Refit each selected model on the complete supplied training period.
5. Report both later test periods separately.

The test periods do not select the model within a sensor combination. They are
reported as an external stability check and must not be used to tune the cost
assumptions, features, models, or threshold.

## Physical sensors and derived features

The source data contains five model features but only four physical sensors:

| Physical sensor | Source feature | Relative cost |
|---|---|---:|
| Light | `Light` | 0.5 |
| Temperature | `Temperature` | 1.0 |
| Humidity | `Humidity` | 1.0 |
| CO2 | `CO2` | 4.0 |

`HumidityRatio` is derived from temperature and relative humidity. It is added
at zero incremental sensor cost whenever both inputs are available; it is
never treated as an independently purchasable sensor.

The costs are illustrative relative points, not vendor prices. They represent
a simplified purchase, integration, calibration, and operating burden. Their
purpose is to make the decision rule explicit and reproducible. A real
procurement recommendation requires sourced prices and cost-sensitivity
analysis.

## Experiment coverage

All 15 non-empty combinations of the four physical sensors are evaluated:

- 4 single-sensor configurations;
- 6 two-sensor configurations;
- 4 three-sensor configurations;
- 1 four-sensor configuration.

The versioned definition is
[`configs/sensor_budget.json`](../configs/sensor_budget.json). The experiment
records its SHA-256 checksum in the generated metadata so results can be tied
to the exact assumptions that produced them.

## Run the experiment

With the package installed:

```powershell
python -m sensorbudget.modeling.sensor_budget
```

With the local `src` workaround:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sensorbudget.modeling.sensor_budget
```

The command performs 375 validation fits:

```text
15 sensor combinations x 5 models x 5 chronological folds
```

It then fits one selected model per sensor combination on all training rows.
Generated artifacts are written to `models/sensor_budget/` and remain outside
Git.

## Generated artifacts

| Artifact | Purpose |
|---|---|
| `sensor_scenarios.csv` | Physical sensors, model features, and assumed cost |
| `cv_fold_metrics.csv` | Metrics for every model, combination, and fold |
| `cv_summary.csv` | Mean and standard deviation across folds |
| `selected_models.csv` | Validation-selected model per combination |
| `heldout_metrics.csv` | Separate metrics for each later test period |
| `heldout_predictions.csv` | Row-level held-out probabilities |
| `metadata.json` | Configuration checksum, environment, and run contract |
| `*.joblib` | Fitted model bundle for each sensor combination |

## Interpretation rule

Cost-performance recommendations must be based on chronological validation,
not on whichever configuration happens to score highest on a held-out test
period. The held-out results answer a different question: whether a
validation-selected configuration remains stable during later periods.

In particular, strong results from Light should be interpreted as evidence
that lighting state is a powerful occupancy proxy in this room. Robustness
tests must still examine daylight, lights left on, automation, and sensor
failure before Light-only deployment could be recommended.

## Cost-scenario sensitivity

After the main experiment, recalculate frontier membership without retraining:

```powershell
python -m sensorbudget.modeling.cost_sensitivity
```

The scenario definitions are versioned in
[`configs/cost_scenarios.json`](../configs/cost_scenarios.json). The command
writes detailed scenario results, frontier rows, membership frequencies, and
configuration checksums under `models/sensor_budget/`.

| Scenario | Temperature | Humidity | Light | CO2 |
|---|---:|---:|---:|---:|
| Current assumptions | 1.0 | 1.0 | 0.5 | 4.0 |
| Equal sensor costs | 1.0 | 1.0 | 1.0 | 1.0 |
| Cheaper CO2 | 1.0 | 1.0 | 0.5 | 1.5 |
| High Light cost (extreme case) | 1.0 | 1.0 | 5.0 | 4.0 |
| High-maintenance CO2 | 1.0 | 1.0 | 0.5 | 8.0 |

Except for the equal-cost scenario, each alternative changes only the sensor
named by the scenario. The other three retain their current assumed costs.
