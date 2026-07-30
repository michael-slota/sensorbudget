# Sensor-budget results

## Experiment

All 15 non-empty combinations of temperature, humidity, light, and CO2 sensors
were evaluated. Five fixed classifiers were compared for each combination
using five expanding chronological folds. The best model for each combination
was selected by mean validation F1 and refitted on the complete training
period. The decision threshold remained at 0.5.

Costs are illustrative relative points rather than market prices. See
[`docs/sensor_budget_experiment.md`](../docs/sensor_budget_experiment.md) for
the full experiment contract and cost assumptions.

## Leading chronological-validation configurations

| Physical sensors | Cost | Selected model | Mean validation F1 |
|---|---:|---|---:|
| Temperature + Light + CO2 | 5.5 | Logistic regression | **0.780** |
| Humidity + Light + CO2 | 5.5 | Random forest | 0.768 |
| Temperature + Humidity + Light + CO2 | 6.5 | Histogram gradient boosting | 0.767 |
| Humidity + Light | 1.5 | Random forest | 0.756 |
| Light | 0.5 | Histogram gradient boosting | 0.755 |
| Temperature + Light | 1.5 | Logistic regression | 0.736 |
| Light + CO2 | 4.5 | Logistic regression | 0.733 |
| Temperature + Humidity + Light | 2.5 | Histogram gradient boosting | 0.731 |

The differences between the leading validation means are small relative to
the known fold-to-fold variability. The all-unoccupied weekend fold gives
every model an F1 of zero, so the fold-level table remains important.

## Held-out observations

These results test later-period stability; they were not used to select the
model within each sensor combination.

| Physical sensors | Cost | Test 1 F1 | Test 2 F1 |
|---|---:|---:|---:|
| Light | 0.5 | 0.971 | 0.982 |
| Temperature + Light | 1.5 | 0.972 | 0.979 |
| Humidity + Light | 1.5 | 0.967 | 0.960 |
| Temperature + Light + CO2 | 5.5 | 0.971 | 0.980 |
| All physical sensors | 6.5 | 0.930 | 0.883 |
| CO2 | 4.0 | 0.827 | 0.449 |
| Temperature + Humidity | 2.0 | 0.770 | 0.654 |

Light-containing configurations dominate these particular periods. The
Light-only model even exceeds the more complex all-sensor baseline on both
held-out periods. This does not establish that Light is intrinsically the best
occupancy sensor: in this dataset, lighting state is nearly a direct proxy for
the label.

## Current conclusion

Three configurations form the validation Pareto frontier under the current
cost assumptions:

| Physical sensors | Cost | Mean validation F1 |
|---|---:|---:|
| Light | 0.5 | 0.755 |
| Humidity + Light | 1.5 | 0.756 |
| Temperature + Light + CO2 | 5.5 | 0.780 |

Humidity + Light is technically Pareto-efficient, but its improvement over
Light alone is only about 0.001 F1. The highest-validation configuration gains
about 0.025 F1 over Light alone at eleven times the assumed cost. These figures
describe the trade-off; they do not establish whether either gain is worth its
cost. That requires cost sensitivity, uncertainty analysis, and an explicit
value assigned to predictive improvement.

Adding Light increases mean validation F1 in all seven matched comparisons,
with gains ranging from about 0.059 to 0.543. Because model selection is
repeated after Light is added, these gains describe the combined sensor-plus-
model pipeline rather than a purely causal Light effect.

A Light-only recommendation remains premature because the dataset does not
cover important lighting failures. Phase 5 should test lights left on while
unoccupied, dark occupancy, sensor loss, noise, and drift before choosing a
deployment configuration.

## Cost-scenario sensitivity

The frontier was recalculated under five illustrative scenarios: current
assumptions, equal sensor costs, cheaper CO2, an extreme high-Light-cost case,
and high-maintenance CO2. Models were not retrained because costs do not alter
the validation predictions.

The cheaper-CO2 scenario changes CO2 from 4.0 to 1.5 while Temperature,
Humidity, and Light remain 1.0, 1.0, and 0.5. The extreme high-Light-cost case
changes Light from 0.5 to 5.0 while the other current assumptions remain
unchanged. The high-maintenance scenario changes only CO2, from 4.0 to 8.0.

| Physical sensors | Frontier scenarios | Share |
|---|---:|---:|
| Light | 5 of 5 | 100% |
| Humidity + Light | 5 of 5 | 100% |
| Temperature + Light + CO2 | 5 of 5 | 100% |
| Temperature | 1 of 5 | 20% |
| CO2 | 1 of 5 | 20% |

The original three frontier points remain efficient in every tested scenario.
Temperature and CO2 enter only in the deliberately extreme high-Light-cost
case. This supports stability across the named assumptions, but
does not cover all possible prices, shared sensor hardware, volume discounts,
or non-additive installation and maintenance costs.

## Limitations

- Relative costs are scenario assumptions and require sensitivity analysis.
- Results come from one room over short collection periods.
- Small differences in mean F1 are uncertain given the fold variability.
- The threshold remains fixed at 0.5.
- Model hyperparameters are fixed rather than tuned.
- The supplied held-out periods are now reported across many predefined
  configurations and should not be repeatedly reused for further tuning.
- Light may encode occupant behavior or office policy rather than occupancy
  itself.
