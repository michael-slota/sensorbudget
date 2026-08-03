# Light-health detector and routing results

> **Project status:** This report records the causal-detector step of Phase 5.
> The planned fault-aware-training comparison is complete; see the
> [fault-aware report](fault_aware_results.md) and consolidated
> [model card](model_card.md).

**Traceability:** Settings are versioned in
[`configs/fault_detection.json`](../configs/fault_detection.json), the
presentation is in
[`notebooks/06_fault_detection_and_routing.ipynb`](../notebooks/06_fault_detection_and_routing.ipynb),
and source identity is fixed by
[`data/source_checksums.json`](../data/source_checksums.json). The Git commit
containing this report identifies the corresponding code revision.

## Selected detector

Training-only chronological validation selected:

| Parameter | Selected value |
|---|---:|
| Trailing stuck window | 20 minutes |
| Allowed stuck-window variation | 0 lx |
| Plausible Light range | Training 0.1st–99.9th percentiles |
| Abrupt-change threshold | Maximum absolute training change |

Mean training-validation results were detection precision 0.377, recall 0.464,
F1 0.398, false-positive rate 0.011, and detection delay 17.2 rows. The modest
score reflects the deliberate inclusion of difficult stuck-low and gradual
bias scenarios rather than only easily detected missing values.

## Held-out detection

| Scenario | Test 1 recall | Test 2 recall | Interpretation |
|---|---:|---:|---|
| Missing | 1.000 | 1.000 | Detected immediately |
| Out of range high | 1.000 | 1.000 | Detected immediately |
| Frozen at current value | 0.171 | 0.000 | Often indistinguishable from natural stability |
| Fixed at high training value | 0.342 | 0.342 | Detected after enough constant observations |
| Fixed at low training value | 0.000 | 0.000 | Deliberately not flagged because normal darkness is constant |
| Negative linear bias | 0.492 | 0.942 | Often crosses the low range threshold |
| Positive linear bias | 0.000 | 0.000 | Remains within plausible held-out range |

The false-positive rate rises from 1.15% in training validation to roughly
3.1% on Test 1 and 3.4–3.6% on Test 2. Consequently, precision is low for short
fault episodes: many detector alerts occur outside the injected interval. This
is evidence of temporal distribution shift in Light behavior.

## Routing effect

For 15- and 60-minute episodes, full-period F1 changes are necessarily smaller
than in the earlier whole-period fault experiment. Important examples are:

| Scenario | Split | Primary only | Detector routing | Oracle routing |
|---|---|---:|---:|---:|
| Missing | Test 1 | 0.956 | 0.971 | 0.971 |
| Out of range high | Test 1 | 0.968 | 0.970 | 0.970 |
| Fixed high | Test 1 | 0.955 | 0.965 | 0.971 |
| Missing | Test 2 | 0.980 | 0.976 | 0.980 |
| Fixed high | Test 2 | 0.975 | 0.973 | 0.980 |

The detector recovers the oracle result for Test 1 missing episodes. In Test 2,
short injected episodes often have little effect on the full-period primary
score, while false alarms route clean observations to the weaker fallback and
slightly reduce F1.

The saved results also include a clean-data control with no injected episode.
For that control, any fallback routing is necessarily a false alarm. This makes
the cost of the health rules independently auditable rather than inferring it
only from fault-containing scenarios.

## Conclusion — no reliability recommendation yet

Simple causal rules are effective for explicit missingness and extreme values,
but they do not reliably solve plausible stuck-at-darkness or gradual positive
calibration bias. A safe detector must accept that some faults are unobservable
from Light alone. The subsequent comparison tests fault-aware training and
missingness indicators in
[`reports/fault_aware_results.md`](fault_aware_results.md).
