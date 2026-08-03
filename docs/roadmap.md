# Project roadmap

This roadmap prioritizes a credible experimental story over a large number of
models. Each phase has a clear deliverable and exit criterion.

## Phase 0 — Project definition

**Status:** Complete

Deliverables:

- Define the decision problem and target.
- Establish repository, data, and documentation conventions.
- Define evaluation and leakage controls.
- Create an experiment catalog.

Exit criterion: another contributor can explain the goal, proposed experiments,
and success criteria from the documentation alone.

## Phase 1 — Reproducible data foundation

**Suggested duration:** 1–2 days

**Status:** Complete

Tasks:

- Run `notebooks/01_data_audit.ipynb` as the initial source-file inspection.
- Implement dataset download or documented manual acquisition.
- Save provenance and SHA-256 checksums.
- Validate schema, ranges, timestamps, duplicates, and target values.
- Combine the supplied files while preserving their source split.
- Produce a concise data-quality report.
- Add unit tests for parsing and validation.

Deliverables:

- `src/sensorbudget/data/load.py`
- `src/sensorbudget/data/validate.py`
- `src/sensorbudget/data/provenance.py`
- `src/sensorbudget/data/build.py`
- `reports/data_quality.md`
- automated data-contract tests

Exit criterion: one command reproduces validated data from the raw inputs.

## Phase 2 — Time-aware exploratory analysis

**Suggested duration:** 2–3 days

**Status:** Complete. The time-aware audit, data-quality report, and proxy-risk
analysis are documented in `notebooks/01_data_audit.ipynb` and
`reports/data_quality.md`. Plotting code needed by the final dashboard will be
extracted during Phase 7 rather than treated as unfinished EDA.

Questions:

- How imbalanced is occupancy?
- How do sensors change before, during, and after occupancy transitions?
- Are supplied train and test periods drawn from comparable distributions?
- Is humidity ratio redundant with temperature and humidity?
- How dominant is the light signal?
- Are there schedule patterns encoded in timestamps?

Suggested figures:

- sensor time series with occupancy overlays;
- class-conditional distributions;
- correlation matrix;
- transition-centered plots;
- train/test drift plots;
- missingness and range summary.

Deliverables:

- `notebooks/01_data_audit.ipynb` (complete);
- `reports/data_quality.md` (complete);
- documented class balance, temporal behavior, split drift, redundancy, and
  Light proxy risk (complete).

Exit criterion: the analysis identifies likely leakage, drift, and proxy-feature
risks before model selection begins.

## Phase 3 — Leakage-safe baselines

**Suggested duration:** 2–3 days

**Status:** Complete. The leakage-safe comparison, chronological model
selection, held-out evaluation, packaged training command, saved artifacts,
and explanatory notebook are implemented. The final model card belongs to
Phase 7, after reliability and operating-threshold decisions. Transition-level
error analysis belongs to Phase 6.

Tasks:

- Add a dummy baseline and interpretable learned baseline.
- Train logistic regression with a fitted preprocessing pipeline.
- Compare decision tree, random forest, and gradient boosting.
- Use chronological validation for tuning.
- Evaluate once on each held-out supplied test period.
- Save configurations, metrics, predictions, and model metadata.

Deliverables:

- reproducible training command;
- baseline comparison table;
- confusion matrices and precision-recall curves;
- `notebooks/02_baseline_models.ipynb`;
- `reports/baseline_results.md`.

Implementation details are documented in `docs/model_training.md`; measured
results are documented in `reports/baseline_results.md`.

Exit criterion: all models are compared on identical splits and the final test
data has not influenced tuning.

## Phase 4 — Sensor-budget experiments

**Suggested duration:** 3–4 days

**Status:** Complete; all 15 physical-sensor combinations and five explicit
cost scenarios have reproducible measured results, and the static README
Pareto chart is published. Phase 4 intentionally makes no final sensor
recommendation.

Run the predefined ablations:

1. All sensors.
2. All except light.
3. Temperature, humidity, and CO2.
4. Temperature and humidity.
5. CO2 only.
6. Light only.
7. Every individual sensor.
8. Selected two- and three-sensor combinations.

Add a simple sensor-cost table using clearly labeled assumptions. Plot
predictive quality against cost and identify the Pareto frontier.

Deliverables:

- ablation results table (complete);
- `notebooks/03_sensor_budget_analysis.ipynb` (complete);
- cost-scenario sensitivity analysis (complete);
- performance-versus-cost chart (complete);
- decision-neutral summary of frontier configurations and trade-offs
  (complete).

Exit criterion: the project quantifies what is lost when each sensor is
removed and identifies shortcut risks to test in Phase 5 without selecting a
deployment configuration.

## Phase 5 — Robustness and reliability

**Suggested duration:** 3–5 days

**Status:** In progress. The reproducible fault-injection comparison,
Light-independent oracle fallback, and training-validated causal Light-health
detector are complete. Fault-aware retraining and the reliability conclusion
remain.

Simulate:

- random missing readings at increasing rates;
- Gaussian noise proportional to each sensor's observed spread;
- a stuck-at-constant sensor;
- gradual calibration drift;
- full loss of one sensor;
- light remaining on in an unoccupied room.

Compare native handling, imputation, missingness indicators, and retraining with
fault augmentation.

Deliverables:

- robustness test harness (complete);
- initial degradation curves (complete);
- initial failure-mode table (complete);
- training-selected Light-independent fallback comparison (complete);
- oracle-gated mitigation analysis (complete);
- `notebooks/04_robustness.ipynb` (complete);
- `notebooks/05_fallback_mitigation.ipynb` (complete);
- Light-health detector and non-oracle routing evaluation (complete);
- false-alarm and missed-fault analysis (complete);
- `notebooks/06_fault_detection_and_routing.ipynb` (complete);
- fault-aware retraining comparison;
- reliability recommendation.

Exit criterion: the candidate system has documented behavior under every
defined sensor fault, including the effect of imperfect fault detection and
routing.

## Phase 6 — Decision and explainability layer

**Suggested duration:** 2–3 days

Tasks:

- Define costs for false occupied and false unoccupied decisions.
- Tune the decision threshold on validation data.
- Check probability calibration.
- Explain global behavior with coefficients or permutation importance.
- Explain representative correct predictions and errors.
- Review false negatives around occupancy transitions.

Deliverables:

- threshold/cost analysis;
- calibration plot;
- global and local explanations;
- documented operating point.

Exit criterion: the selected threshold is justified by an explicit operational
objective rather than the default value of 0.5.

## Phase 7 — Packaging and communication

**Suggested duration:** 2–4 days

Build a static Plotly dashboard suite for GitHub Pages using precomputed,
versioned results. Retain the packaged experiment and batch-scoring commands
for reproducibility. A server-hosted Plotly Dash application may remain an
optional extension, but it is not required for the GitHub Pages release.

Complete:

- final model card;
- reproducibility instructions;
- architecture diagram;
- polished report;
- overview, EDA, model, sensor-budget, robustness, and mitigation pages;
- interactive static Plotly model and sensor-subset comparison views;
- methodology, limitations, dataset attribution, and license notices;
- GitHub Pages deployment workflow;
- CI for linting and tests;
- optional server-hosted Dash application and container image.

Exit criterion: a new user can inspect the published dashboard, understand the
main evidence and limitations, install the project, and reproduce the
headline results from documented commands.

## Stretch goals

These should begin only after the core classification study is complete:

- Predict occupancy 5, 15, and 30 minutes ahead.
- Add causal caution around time-of-day and light proxies.
- Detect occupancy transitions as events rather than classifying every row.
- Create an online-learning drift monitor.
- Estimate ventilation-energy savings under different policies.
- Test generalization on a second building or occupancy dataset.

## Suggested headline outputs

For the final standalone project presentation, prioritize:

1. A timeline showing sensor behavior and occupancy transitions.
2. A model comparison table using chronological evaluation.
3. A sensor-ablation heatmap.
4. A performance-versus-cost Pareto chart.
5. Robustness degradation curves.
6. A threshold trade-off tied to operational costs.
7. A short, honest limitations section.
