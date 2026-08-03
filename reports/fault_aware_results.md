# Fault-aware training results

> **Project status:** This report completes the Phase 5 mitigation comparison.
> Detector routing is the strongest tested strategy on average, but the project
> does not certify it for deployment. See the consolidated
> [model card](model_card.md).

**Traceability:** Settings are versioned in
[`configs/fault_aware_training.json`](../configs/fault_aware_training.json),
the presentation is in
[`notebooks/07_fault_aware_training.ipynb`](../notebooks/07_fault_aware_training.ipynb),
and source identity is fixed by
[`data/source_checksums.json`](../data/source_checksums.json). The Git commit
containing this report identifies the corresponding code revision.

## Training-validation selection

Both representations selected logistic regression with a 1% added fault
sample. The plain representation reached mean clean CV F1 0.760 versus 0.780
for the clean-trained reference, missing the 0.02 guardrail by approximately
0.00003; it is retained only as a diagnostic candidate. The missing-indicator
representation reached 0.761 and satisfied the guardrail. Their mean
whole-period fault CV F1 values were both approximately 0.432.

## Clean held-out behavior

| Strategy | Test 1 F1 | Test 2 F1 |
|---|---:|---:|
| Existing primary | 0.971 | 0.980 |
| Detector routing | 0.971 | 0.976 |
| Fault-aware, no indicator (diagnostic) | 0.971 | 0.917 |
| Fault-aware + missing indicator | 0.971 | 0.942 |

The training-validation guardrail did not guarantee later-period stability.
The accepted missing-indicator candidate retained Test 1 performance but lost
0.038 F1 on pristine Test 2. The plain diagnostic candidate lost 0.063.

## Faulted held-out behavior

The missingness indicator prevents the Test 1 missing-Light loss that remains
in the plain fault-aware model. It does not identify plausible stuck values or
gradual calibration bias, because those readings are present rather than
missing. Across all clean and episodic held-out cases, detector routing has the
highest mean F1 on both Test 1 (0.970) and Test 2 (0.974). The missing-indicator
model averages 0.969 and 0.939 respectively.

## Reliability conclusion

The tested fault-aware retraining schemes should not replace the current
primary-plus-detector-routing design. Sparse augmentation can teach an explicit
missingness response, but it introduces a substantial Test 2 clean-data cost
and does not solve faults that remain observationally plausible. Detector
routing also has known missed-fault and false-alarm limitations, so this is a
comparative project conclusion rather than a deployment certification or final
sensor recommendation.

