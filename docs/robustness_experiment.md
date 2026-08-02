# Robustness experiment

## Purpose

Phase 5 tests how the validation-frontier configurations behave when sensor
readings no longer follow the clean data-generating conditions. The comparison
includes:

- Light;
- Humidity + Light;
- Temperature + Light + CO2.

These configurations are comparison cases, not deployment finalists. The
experiment does not make a sensor recommendation.

## Evaluation contract

The robustness command loads the fitted model bundles created by the
sensor-budget experiment and evaluates perturbed copies of Test 1 and Test 2.
Models are not retrained.

Training-period data is used only to calculate:

- medians for missing-value fallback;
- standard deviations for noise and drift severity;
- observed ranges for bounded measurement noise;
- low and high quantiles for stuck sensors;
- representative occupied and unoccupied Light values.

Every metric is compared with the same fitted model's clean held-out baseline.
Random missingness and Gaussian noise use five deterministic repetitions per
severity.

## Scenarios

| Group | Intervention |
|---|---|
| Light policy | Assign occupied-like Light to unoccupied rows |
| Light policy | Assign unoccupied-like Light to occupied rows |
| Random missing | Remove 1%, 5%, 10%, 20%, or 40% of feature cells |
| Gaussian noise | Add 0.1, 0.25, 0.5, or 1.0 training standard deviations |
| Complete loss | Replace one unavailable sensor with its training median |
| Stuck sensor | Freeze one sensor at its training 5th or 95th percentile |
| Gradual drift | Ramp one sensor to ±0.5 or ±1.0 training standard deviations |

Randomly missing values and complete loss use training-median fallback. Noise
is clipped to the training-period observed range. Drift is not clipped because
out-of-range calibration movement is part of the failure being tested.

## Diagnostic Light interventions

The Light-policy scenarios use the true label to decide which rows to modify.
They are diagnostic counterfactuals, not transformations available to a live
prediction system:

- **unoccupied lit:** every unoccupied row receives the median Light value
  observed among occupied training rows;
- **occupied dark:** every occupied row receives the median Light value
  observed among unoccupied training rows.

These deliberately break the Light–occupancy relationship learned from this
room and quantify shortcut reliance.

## Run

Run the sensor-budget experiment first so fitted bundles exist, then execute:

```powershell
python -m sensorbudget.robustness.evaluate
```

With the local source-path workaround:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sensorbudget.robustness.evaluate
```

Generated artifacts are written to `models/robustness/`:

| Artifact | Purpose |
|---|---|
| `robustness_metrics.csv` | Scenario metrics and deltas from clean baseline |
| `metadata.json` | Configuration hashes, statistics, and run contract |

## Interpretation limits

- The scenarios are controlled simulations, not measured hardware failures.
- Label-targeted Light interventions are intentionally severe diagnostics.
- Both held-out periods come from the same room.
- The same held-out periods have already been reported in earlier phases, so
  these results must not be used for repeated model or threshold tuning.
- Median fallback is a basic mitigation, not necessarily the best missing-data
  strategy.
- Reliability conclusions do not establish cross-building generalization.
