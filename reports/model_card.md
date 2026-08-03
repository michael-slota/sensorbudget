# Model card: SensorBudget occupancy research candidate

## Executive summary

The project’s main research candidate is a class-balanced logistic regression
using Temperature, Light, and CO2. At threshold `0.50`, clean F1 is `0.971` on
Test 1 and `0.980` on Test 2. Those scores are not sufficient evidence for a
deployment decision:

- complete Light loss and occupied darkness reduce F1 to approximately zero;
- detector routing provides the strongest tested mitigation overall but has
  known false alarms and missed plausible faults;
- a validation-selected threshold of `0.86` performs worse than `0.50` under
  the same illustrative cost objective on both held-out periods; and
- the data covers one office over a short period, with no external building or
  seasonal validation.

The central conclusion is therefore conditional: the candidate performs well
on clean source periods, while reliability depends strongly on lighting
behaviour, temporal context, and fault handling.

## Model-card status

This card documents the main research candidate produced by SensorBudget as of
Phase 6. It records the evidence, assumptions, and limitations in this
repository; it is not a deployment approval or a final sensor recommendation.

| Item | Value |
|---|---|
| Task | Minute-level binary office-occupancy classification |
| Primary candidate | Class-balanced logistic regression |
| Inputs | Temperature, Light, and CO2 |
| Output | Estimated probability of `Occupancy = 1` and a thresholded class |
| Reference threshold | 0.50 for the strongest observed held-out reference |
| Validation-selected threshold | 0.86 under illustrative equal error costs |
| Reliability mitigation studied | Causal Light-health detector routing to a Temperature + CO2 logistic fallback |
| Development status | Research prototype; not deployment-certified |

## Intended use

The candidate is intended to support reproducible research into:

- occupancy detection from indoor environmental sensors;
- time-aware model evaluation;
- performance-versus-sensor-cost comparisons;
- sensitivity to missing, noisy, drifting, and stuck sensors;
- explicit decision-threshold and error-cost analysis; and
- transparent global and row-level model explanations.

It may also serve as a reference implementation for an offline batch-scoring
pipeline. Any use in a real building requires new data, stakeholder-defined
costs, prospective validation, monitoring, and a documented fail-safe policy.

## Out-of-scope and unsuitable uses

The current model should not be used without further validation for:

- safety-critical control, emergency response, or access control;
- monitoring or evaluating individual people;
- buildings, rooms, seasons, lighting policies, or sensor hardware not
  represented in the source data;
- autonomous HVAC or lighting control with no operational override; or
- claims that Light, CO2, or another feature causes occupancy.

The dataset contains room-level measurements rather than personal attributes,
but occupancy inference can still reveal patterns about when people use a
space. A real deployment should define retention, access, aggregation, and
privacy rules before collecting or exposing predictions.

## Data

The project uses the
[UCI Occupancy Detection dataset](https://archive.ics.uci.edu/dataset/357/occupancy%2Bdetection)
(Candanedo, 2016; CC BY 4.0). It contains 20,560 approximately minute-level
observations from one office room:

| Source split | Rows | Period | Role in this project |
|---|---:|---|---|
| Training | 8,143 | 2015-02-04 to 2015-02-10 | Fitting and expanding chronological validation |
| Test 1 | 2,665 | 2015-02-02 to 2015-02-04 | Earlier held-out source period |
| Test 2 | 9,752 | 2015-02-11 to 2015-02-18 | Later held-out source period |

The source files contain Temperature, Humidity, Light, CO2, derived
HumidityRatio, timestamp, row ID, and binary Occupancy. The project verifies
committed SHA-256 checksums before experiments. The raw third-party files are
not committed to this repository.

Important data characteristics are:

- occupied prevalence is 21.2% in training, 36.5% in Test 1, and 21.0% in
  Test 2;
- occupancy is absent overnight and on the observed weekends;
- Light has an approximately 0.915 correlation with occupancy across the
  source periods;
- only one occupied row is completely dark across all 20,560 observations;
- Temperature shifts most strongly in Test 1, while Humidity and
  HumidityRatio shift most strongly in Test 2; and
- the data represents one room over a short period, with no external building
  or seasonal validation.

See [`reports/data_quality.md`](data_quality.md) for the complete audit and
[`data/source_checksums.json`](../data/source_checksums.json) for source
identity.

## Model development

### Selection procedure

Candidate models were compared using five expanding chronological folds within
the supplied training period. Every validation block occurs after its
corresponding development block; adjacent minute-level rows are never randomly
shuffled between them. F1 for the occupied class is the primary selection
metric, with average precision and Brier score used as tie-breakers.

The sensor-budget experiment evaluated all 15 non-empty combinations of the
four physical sensors. Temperature + Light + CO2 with logistic regression had
the highest mean chronological-validation F1 (0.780), compared with 0.755 for
Light alone and 0.767 for all physical sensors. The differences are uncertain
because fold variability is large and one validation fold contains no occupied
rows.

### Primary candidate

The main candidate is a scikit-learn pipeline containing:

1. `StandardScaler`, fitted on development or training rows only; and
2. class-balanced `LogisticRegression(max_iter=1000)`.

It uses Temperature, Light, and CO2. Humidity is not used, and HumidityRatio is
excluded because it is derived from Temperature and Humidity rather than being
an additional physical sensor.

This candidate should not be confused with the initial all-sensor baseline.
The baseline experiment selected histogram gradient boosting using
Temperature, Humidity, Light, CO2, and HumidityRatio and obtained F1 0.930 on
Test 1 and 0.883 on Test 2. The later sensor-budget study selected the simpler
three-sensor logistic candidate for the reliability and explainability work.

## Evaluation results

### Clean source periods at threshold 0.50

| Evaluation period | F1 | Interpretation |
|---|---:|---|
| Test 1 | 0.971 | Strong performance on the earlier source period |
| Test 2 | 0.980 | Strong performance on the later source period |

These results are useful comparisons, not estimates of generalization to other
rooms. Although the tests did not select the original model, both periods were
examined repeatedly in later ablation, robustness, routing, and threshold
analyses. They must therefore not be treated as fresh confirmation data for
future model changes.

### Sensor-budget evidence

Under illustrative relative costs, Light, Humidity + Light, and Temperature +
Light + CO2 form the validation Pareto frontier in all five tested cost
scenarios. The cost values are sensitivity assumptions rather than purchase,
installation, energy, or maintenance prices. The experiment deliberately makes
no final sensor recommendation.

### Robustness evidence

Clean accuracy hides a shared dependency on Light:

- simulated occupied darkness and complete Light loss reduce the primary
  model's F1 to approximately zero;
- unoccupied-but-lit behavior reduces F1 to roughly 0.53 on Test 1 and 0.35 on
  Test 2;
- 40% random missing feature cells with training-median replacement reduce
  mean F1 to roughly 0.71–0.74 across the studied frontier configurations;
- strong noise and gradual Light drift cause period-dependent degradation; and
- adding sensors does not create automatic fallback behavior when the fitted
  model still relies on Light.

A Temperature + CO2 logistic fallback recovers F1 to 0.817 on Test 1 and 0.540
on Test 2 when an oracle identifies severe Light failures. A causal detector
can recognize explicit missingness and extreme values, but it misses plausible
stuck-dark and positive-bias faults and produces roughly 3–3.6% false-positive
alerts on held-out data. Detector routing is the strongest mitigation tested
on average, but it remains a research design with missed faults and unnecessary
routing.

Fault-aware retraining with a missingness indicator was not accepted as a
replacement: it retained Test 1 F1 but reduced pristine Test 2 F1 from 0.980
to 0.942 and did not solve plausible present-but-wrong readings.

### Threshold and calibration evidence

Under an illustrative equal-cost objective, chronological validation selected
a threshold of 0.86 and reduced assumed validation error cost from 11.05 to
5.75 units per 1,000 rows. That apparent advantage did not transfer:

| Period | Cost/1,000 at 0.50 | Cost/1,000 at 0.86 | Effect of 0.86 |
|---|---:|---:|---|
| Test 1 | 21.39 | 22.51 | Slightly worse |
| Test 2 | 8.41 | 22.05 | Materially worse |

On Test 2, threshold 0.86 reduces recall from 0.994 to 0.918 and F1 from 0.980
to 0.946. The project therefore records 0.86 as the auditable
validation-selected threshold but does not recommend deploying it. Threshold
0.50 is the stronger observed held-out reference; it is not re-selected or
declared universally optimal from test results.

Calibration diagnostics are also period-dependent:

| Evaluation data | Brier score | Expected calibration error |
|---|---:|---:|
| Chronological validation | 0.0107 | 0.0274 |
| Test 1 | 0.0189 | 0.0123 |
| Test 2 | 0.0103 | 0.0219 |

No post-hoc calibration transform is fitted because a credible comparison
would require an additional nested temporal validation procedure.

## Explainability

For the standardized logistic model, Light has the largest positive
coefficient (+4.50), followed by CO2 (+1.86); Temperature has a negative
conditional coefficient (-1.23). Local explanations decompose each row's
log-odds into the intercept and its three feature contributions.

These explanations describe the fitted model, not physical causation. Their
strong Light contribution corroborates the robustness finding that the model
has learned the room's lighting behavior as a proxy for occupancy.

At threshold 0.86, Test 1 recall rises from 0.944 during the first 0–2 minutes
after occupancy onset to 0.999 after 15 minutes. Test 2 recall begins at 0.857
and remains only 0.918 after 15 minutes, indicating broader temporal shift
rather than only a short response delay.

## Limitations and risks

- **External validity:** one room, one building context, and a short winter
  collection period cannot establish performance elsewhere.
- **Proxy dependence:** Light may encode manual lighting policy, daylight, or
  occupant routine rather than occupancy itself.
- **Temporal instability:** feature distributions, detector false alarms,
  fallback quality, and the validation-selected threshold change across source
  periods.
- **Limited model search:** model hyperparameters are fixed baselines rather
  than the result of extensive tuning.
- **Synthetic faults:** injected faults diagnose sensitivity but are not
  estimates of real hardware failure frequency or severity.
- **Test reuse:** predefined held-out periods support transparent diagnostics,
  but repeated examination means new development should be confirmed on fresh
  rooms and time periods.
- **Assumed economics:** relative sensor costs and false-positive/false-negative
  costs are illustrative. No measured comfort, energy, installation, or
  maintenance economics are available.
- **No deployment policy:** alert handling, manual overrides, abstention,
  retraining, monitoring, privacy, and fail-safe behavior remain unspecified.

## Recommended validation before deployment

1. Collect prospective data across multiple rooms, seasons, daylight
   conditions, occupancy patterns, and sensor units.
2. Freeze the complete model, threshold, detector, and routing policy before
   evaluating on that data.
3. Obtain stakeholder-owned costs for missed occupancy and unnecessary
   occupied responses.
4. Test real fault logs and measure detection precision, recall, delay, false
   alarms, and downstream control impact.
5. Define monitoring for feature ranges, missingness, probability drift,
   calibration, occupancy prevalence, and fallback usage.
6. Establish privacy, retention, access-control, human-override, and rollback
   procedures.

## Reproduction and traceability

The results summarized here are produced by the packaged commands documented
in the repository [`README.md`](../README.md). Experiment assumptions are
versioned under [`configs/`](../configs), implementation code is under
[`src/sensorbudget/`](../src/sensorbudget), and detailed evidence is recorded
in the numbered notebooks and narrative reports.

The source data are verified against
[`data/source_checksums.json`](../data/source_checksums.json). Generated model
bundles and row-level artifacts are intentionally excluded from Git and can be
recreated by running the documented pipeline. Software and original
documentation are MIT-licensed; the UCI data remain subject to CC BY 4.0.

## Evidence index

- [`reports/data_quality.md`](data_quality.md): source audit and EDA
- [`reports/baseline_results.md`](baseline_results.md): initial model comparison
- [`reports/sensor_budget_results.md`](sensor_budget_results.md): sensor subsets and cost scenarios
- [`reports/robustness_results.md`](robustness_results.md): fault injection
- [`reports/fallback_mitigation_results.md`](fallback_mitigation_results.md): oracle fallback
- [`reports/fault_detection_results.md`](fault_detection_results.md): causal detector and routing
- [`reports/fault_aware_results.md`](fault_aware_results.md): fault-aware retraining
- [`reports/decision_explainability_results.md`](decision_explainability_results.md): threshold, calibration, and explanations
