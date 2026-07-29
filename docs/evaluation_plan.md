# Evaluation plan

## Decision problem

Given sensor information available at time \(t\), predict whether the room is
occupied at time \(t\). Forecasting future occupancy is a separate stretch
task and must use shifted targets.

## Split strategy

Use the source-provided training and test periods as the primary split. Within
the training period, use expanding-window or blocked time-series validation for
model and threshold selection.

Random row-level cross-validation may be shown only as a demonstration of
optimistic leakage; it must not support the headline result.

## Metrics

Primary metric:

- F1 for the occupied class, until operational costs are defined.

Required supporting metrics:

- precision and recall;
- balanced accuracy;
- PR-AUC / average precision;
- ROC-AUC;
- confusion matrix;
- Brier score and calibration curve;
- false-negative rate around occupancy transitions.

Report a bootstrap confidence interval using contiguous time blocks where
practical. Ordinary row-wise bootstrap intervals would ignore autocorrelation.

## Model-selection protocol

1. Freeze test periods.
2. Develop preprocessing and features on training data.
3. Tune models using time-aware validation.
4. Select the sensor set, model, and threshold using validation results.
5. Evaluate the chosen specification on held-out tests.
6. Report all material deviations from the protocol.

## Operational evaluation

Once cost assumptions are defined, compute:

```text
expected_cost =
    false_occupied_count * false_occupied_cost
  + false_unoccupied_count * false_unoccupied_cost
  + sensor_cost
  + inference_or_maintenance_cost
```

Use multiple scenarios if precise costs are unavailable. Do not present invented
sensor prices as measured facts.

## Robustness acceptance criteria

Initial proposed targets, to be revised after baseline measurement:

- F1 degradation below 5 percentage points with 10% random missingness.
- No catastrophic failure when any single non-light sensor is unavailable.
- A documented fallback when light is stuck high.
- Stable conclusions across both supplied test periods.

## Reproducibility record

Every result should record:

- Git commit;
- configuration;
- random seed;
- Python and package versions;
- raw-data checksums;
- split boundaries;
- selected threshold;
- metrics and generated artifact paths.

