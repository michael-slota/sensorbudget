# Light-independent fallback results

> **Project status:** This report records the oracle-routing step of Phase 5.
> The planned causal detector and routing evaluation is complete; see the
> [fault-detection report](fault_detection_results.md) and consolidated
> [model card](model_card.md).

**Traceability:** Settings are versioned in
[`configs/fallback_mitigation.json`](../configs/fallback_mitigation.json), the
presentation is in
[`notebooks/05_fallback_mitigation.ipynb`](../notebooks/05_fallback_mitigation.ipynb),
and source identity is fixed by
[`data/source_checksums.json`](../data/source_checksums.json). The Git commit
containing this report identifies the corresponding code revision.

## Scope

This experiment evaluates whether known Light faults can be mitigated by
routing predictions from the Temperature + Light + CO2 primary model to a
Light-independent fallback. It does not select a final sensor configuration or
claim that faults can already be detected in production.

## Training-only selection

Among eligible models using sensors already present in the primary system,
Temperature + CO2 logistic regression had the highest chronological
cross-validation mean F1:

| Fallback candidate | Selected model | Mean validation F1 |
|---|---|---:|
| Temperature + CO2 | Logistic regression | 0.679 |
| CO2 | Logistic regression | 0.674 |
| Temperature | Histogram gradient boosting | 0.408 |

Test 1 and Test 2 were not used for this selection.

## Held-out results

| Condition | Test 1 primary | Test 1 oracle fallback | Test 2 primary | Test 2 oracle fallback |
|---|---:|---:|---:|---:|
| Clean reference | 0.971 | 0.817 | 0.980 | 0.540 |
| Complete Light loss | 0.000 | 0.817 | 0.001 | 0.540 |
| Light stuck low | 0.000 | 0.817 | 0.001 | 0.540 |
| Light stuck high | 0.535 | 0.817 | 0.347 | 0.540 |
| Unoccupied but lit | 0.534 | 0.817 | 0.347 | 0.540 |
| Occupied but dark | 0.000 | 0.817 | 0.001 | 0.540 |

F1 is shown at the existing 0.5 classification threshold.

## Interpretation

Oracle routing recovers substantial predictive ability during severe Light
failures. Under complete loss, the F1 gain is 0.817 on Test 1 and 0.539 on Test
2 relative to leaving the faulted value with the primary model.

The recovery is incomplete and unstable across periods. The fallback is 0.155
below the clean primary on Test 1 and 0.440 below it on Test 2. This difference
is evidence that non-Light relationships generalize less consistently.

Routing is also not automatically beneficial for gradual drift. The primary
model remains better than the fallback for every tested drift level, including
the strongest shifts. A useful router must therefore distinguish catastrophic
or policy-related failure from mild calibration drift rather than switching on
every anomaly.

## Conclusion — no final recommendation

A Light-independent fallback is a credible mitigation for known severe Light
faults, but the oracle experiment does not solve fault detection and does not
maintain clean-model performance. The subsequent causal-routing experiment
measures false alarms, missed faults, and end-to-end gated performance in
[`reports/fault_detection_results.md`](fault_detection_results.md).
