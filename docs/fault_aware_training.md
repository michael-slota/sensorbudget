# Fault-aware training experiment

This experiment tests whether exposing an occupancy classifier to simulated
Light faults during training is a safer mitigation than switching to a
Light-independent fallback model at prediction time.

## Leakage controls

- Model algorithm and augmentation strength are selected only with five
  expanding chronological folds of the supplied training period.
- Light reference values and imputation medians are recalculated from each
  fold's development rows.
- Test 1 and Test 2 are used only after the choices are fixed.
- Held-out fault episodes use the same deterministic placements as the causal
  detector experiment, enabling direct strategy comparisons.

## Training representations

Both representations use Temperature, Light, and CO2:

1. `fault_aware` replaces missing Light with the training median.
2. `fault_aware_missing_indicator` performs the same replacement and adds a
   binary `Light_missing` feature. The indicator is calculated before
   replacement, so the model can distinguish an observed median-like value
   from an unavailable measurement.

Every development fold retains all original rows and adds a fault-augmented
sample containing equal shares of missing, frozen, fixed-low, fixed-high,
out-of-range, positive-bias, and negative-bias Light examples. Added sample
sizes of 1%, 5%, and 10% are compared. Logistic regression, decision tree,
random forest, and histogram gradient boosting are evaluated at each strength.

## Selection rule

The clean-trained Temperature + Light + CO2 logistic regression is the clean
reference. A fault-aware candidate is eligible only when its mean clean CV F1
is no more than 0.02 below that reference. Eligible candidates maximize the
equally weighted average of clean F1 and mean F1 across all seven whole-period
validation faults.

If no candidate satisfies the guardrail, the least damaging candidate is saved
only for diagnosis and explicitly marked as rejected. This avoids silently
weakening the reliability requirement to manufacture a positive result.

## Reproduction

Run the sensor-budget and fault-detection stages first, then:

```powershell
python -m sensorbudget.robustness.fault_aware
```

The command writes CV details, candidate summaries, held-out metrics, fitted
diagnostic bundles, and metadata to `models/fault_aware/`. Generated model
artifacts are excluded from version control.

