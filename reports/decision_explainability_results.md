# Decision threshold and explainability results

## Executive summary

- Equal-cost chronological validation selects threshold `0.86`, reducing its
  assumed cost from `11.05` to `5.75` per 1,000 validation rows.
- That advantage does not transfer: threshold `0.50` has lower equal-cost error
  on both held-out periods, especially Test 2.
- Light has the largest standardized model coefficient, reinforcing the
  robustness finding that the candidate relies strongly on lighting behaviour.

> **Project status:** This report completes Phase 6. It documents an auditable
> validation-selected threshold but makes no unconditional operating-point or
> deployment recommendation. See the consolidated [model card](model_card.md).

**Traceability:** Settings are versioned in
[`configs/decision_explainability.json`](../configs/decision_explainability.json),
the presentation is in
[`notebooks/08_decision_explainability.ipynb`](../notebooks/08_decision_explainability.ipynb),
and source identity is fixed by
[`data/source_checksums.json`](../data/source_checksums.json). The Git commit
containing this report identifies the corresponding code revision.

## Validation-selected operating points

| Cost scenario | Selected threshold | Cost/1,000 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Equal error cost | 0.86 | 5.75 | 0.974 | 0.996 | 0.985 |
| Comfort-focused | 0.86 | 8.70 | 0.974 | 0.996 | 0.985 |
| Energy-focused | 0.88 | 23.58 | 0.977 | 0.988 | 0.983 |

The equal-cost 0.86 threshold reduces validation cost from 11.05 at the default
0.5 threshold to 5.75 per 1,000 rows. The probabilities are highly separated,
so raising the threshold removes false positives while preserving validation
recall.

## Held-out confirmation

The validation advantage does not transfer uniformly. Under equal costs, Test
1 cost is 22.51 per 1,000 rows at 0.86 versus 21.39 at 0.5. Test 2 cost rises
from 8.41 to 22.05 because recall falls from 0.994 to 0.918. This is evidence
of threshold instability under temporal distribution shift. The project keeps
0.86 as the validation-selected reference for auditability, but does not claim
it is deployment-ready.

## Calibration

| Dataset | Brier score | Expected calibration error |
|---|---:|---:|
| Chronological validation | 0.0107 | 0.0274 |
| Test 1 | 0.0189 | 0.0123 |
| Test 2 | 0.0103 | 0.0219 |

The raw probabilities have low aggregate Brier error, but calibration varies
across time and should not be interpreted as exact occupancy frequencies. No
post-hoc recalibration is applied in this phase.

## Explanations and transition behavior

Standardized coefficients rank Light first (+4.50), CO2 second (+1.86), and
Temperature third with a negative association (-1.23), conditional on the
other two features. These are associations in the fitted model, not causal
effects.

At threshold 0.86, Test 1 recall is lowest in the first 0–2 minutes after an
occupancy onset (0.944) and reaches 0.999 after 15 minutes. Test 2 differs:
recall is 0.857 in the first 0–2 minutes and remains only 0.918 after 15
minutes. The later-period false negatives are therefore not solely a short
transition-delay problem.

## Operating-point conclusion

Phase 6 documents 0.86 as the threshold selected by the predefined equal-cost
validation objective. Held-out evidence shows that this threshold is not
stable enough for an unconditional deployment recommendation. A real operating
point requires domain-owned error costs and prospective monitoring or broader
building data.

