# Light-health detection and realistic routing

## Objective

The oracle-fallback experiment assumed perfect knowledge of every Light fault.
This experiment replaces that assumption with causal rules that see only the
current and previous Light readings. It asks whether observable technical
faults can be detected accurately enough to route occupancy predictions from
the existing Temperature + Light + CO2 primary model to the existing
Temperature + CO2 fallback.

Neither occupancy model is retrained in this phase.

## Synthetic fault episodes

Faults are inserted into temporary copies of otherwise clean chronological
periods. Each episode has a known start, end, and row-level fault label:

- `missing`: Light is unavailable;
- `stuck_current`: Light freezes at its value when the episode begins;
- `stuck_low`: Light is fixed at the training 5th percentile;
- `stuck_high`: Light is fixed at the training 95th percentile;
- `out_of_range_high`: Light exceeds the training maximum by one training
  standard deviation;
- `linear_bias_positive`: additive calibration bias grows from zero to +1
  training standard deviation;
- `linear_bias_negative`: additive calibration bias grows from zero to -1
  training standard deviation.

Temperature, CO2, timestamps, and occupancy labels remain unchanged. Original
source files are never modified.

Occupied-dark and unoccupied-lit counterfactuals are not detector targets.
They describe a changed relationship between lighting and occupancy, not
necessarily a technical sensor failure observable from Light alone.

## Causal detector rules

The detector combines four transparent rules:

1. flag a missing current reading;
2. flag a trailing window with no Light variation, except normal low-light
   darkness;
3. flag values outside training-derived quantile bounds;
4. flag a current absolute change larger than a training-derived threshold.

Every rolling calculation is trailing: no future row is used to classify the
current row. Constant darkness is excluded from the stuck rule because about
two-thirds of the source periods contain legitimate extended zero-Light
sequences. This safety choice means a sensor stuck at darkness is difficult to
distinguish without another source of context.

## Training-only selection

Detector settings are selected with five chronological folds inside the
supplied training period. Each candidate is exposed to seven fault types,
15/30/60-minute episodes, and repeated placements.

Candidates must first satisfy a mean validation false-positive rate of at most
5%. The remaining candidate with the highest mean detection F1 is selected;
ties favor fewer false alarms and shorter detection delay. Test 1 and Test 2 do
not influence this choice.

## Routing strategies

For every held-out simulated episode, four systems are compared:

- `primary_only`: always use Temperature + Light + CO2;
- `fallback_only`: always use Temperature + CO2;
- `oracle_routing`: use the fallback exactly on known injected-fault rows;
- `detector_routing`: use the fallback only where the causal detector fires.

The same four strategies are also evaluated on completely clean Test 1 and
Test 2 periods. This control isolates performance lost solely to false alarms.

Missing Light is replaced with the training median before primary prediction,
matching the earlier robustness experiment. The fallback ignores Light.

## Reproduction

Run the Phase 4 sensor-budget experiment first so both fitted bundles exist,
then run:

```powershell
python -m sensorbudget.robustness.fault_detection
```

Generated artifacts are written under `models/fault_detection/`:

- `detector_cv_results.csv`;
- `heldout_detection_metrics.csv`;
- `heldout_routing_metrics.csv`;
- `example_episode_trace.csv`;
- `metadata.json`.

The results notebook is `notebooks/06_fault_detection_and_routing.ipynb`.

## Interpretation boundary

This remains a simulation study. A real building would require hardware fault
logs to validate whether injected episodes resemble actual sensor failures.
The rules detect symptoms, not physical root causes, and their thresholds can
experience distribution shift across periods.
