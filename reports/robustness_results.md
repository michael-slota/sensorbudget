# Robustness results

## Scope

The three validation-frontier configurations were evaluated under predefined
sensor perturbations on Test 1 and Test 2. These are comparison cases rather
than deployment finalists, and no sensor recommendation is made.

The complete experiment contract is documented in
[`docs/robustness_experiment.md`](../docs/robustness_experiment.md).

## Clean baselines

| Physical sensors | Test 1 F1 | Test 2 F1 |
|---|---:|---:|
| Light | 0.971 | 0.982 |
| Humidity + Light | 0.967 | 0.960 |
| Temperature + Light + CO2 | 0.971 | 0.980 |

## Lighting-policy failures

| Scenario | Configuration | Test 1 F1 | Test 2 F1 |
|---|---|---:|---:|
| Unoccupied room appears lit | Light | 0.533 | 0.346 |
| Unoccupied room appears lit | Humidity + Light | 0.530 | 0.345 |
| Unoccupied room appears lit | Temperature + Light + CO2 | 0.534 | 0.347 |
| Occupied room appears dark | Light | 0.000 | 0.000 |
| Occupied room appears dark | Humidity + Light | 0.000 | 0.000 |
| Occupied room appears dark | Temperature + Light + CO2 | 0.000 | 0.001 |

All three models fail under occupied darkness and degrade strongly when
unoccupied rows appear lit. Additional sensors do not provide an automatic
fallback because the fitted models still rely heavily on Light.

## Complete sensor loss

Replacing Light entirely with its training median reduces F1 to approximately
zero for every configuration on both test periods. Loss of Humidity has only a
small effect on Humidity + Light. Within Temperature + Light + CO2, Temperature
loss has a small-to-moderate effect and CO2 loss varies by period, but none of
those additional signals compensates for complete Light loss.

## Random missingness and noise

At 40% randomly missing feature cells with training-median fallback, mean F1
across five repetitions ranges from approximately 0.710 to 0.741 across the
three configurations and two periods.

At Gaussian noise equal to one training standard deviation, mean F1 ranges
from approximately 0.680 to 0.833. The Temperature + Light + CO2 configuration
is most affected on Test 2 at this severity, showing that more inputs do not
guarantee better noise robustness.

## Stuck readings and drift

A low stuck Light value reproduces the occupied-dark failure: F1 falls to
approximately zero for every configuration. Gradual Light drift is less
catastrophic but remains material. Under the worst tested drift direction,
Light-only falls to F1 0.781 on Test 1 and 0.634 on Test 2; the multi-sensor
models also degrade, especially on Test 2.

## Findings — no sensor recommendation

- Clean held-out performance hides a shared dependence on Light.
- Adding sensors did not create fault tolerance in the current fitted models.
- Median fallback limits degradation under scattered missing cells but does
  not solve complete Light loss.
- Severe noise and calibration drift create period-dependent degradation.
- No tested configuration passes the critical Light-failure scenarios.

The next robustness increment should compare mitigation strategies using
training-period validation: explicit missingness indicators, fault-augmented
training, and models trained without Light as a fallback path. These results
do not select or recommend a sensor configuration.
