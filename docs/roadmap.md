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

**Status:** Core EDA complete in `notebooks/01_data_audit.ipynb`; reusable
pipeline extraction remains.

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

- `notebooks/01_data_audit.ipynb`
- `reports/data_quality.md`
- reusable plotting code moved into `src/`

Exit criterion: the analysis identifies likely leakage, drift, and proxy-feature
risks before model selection begins.

## Phase 3 — Leakage-safe baselines

**Suggested duration:** 2–3 days

Tasks:

- Add dummy and rule-based baselines.
- Train logistic regression with a fitted preprocessing pipeline.
- Compare decision tree, random forest, and gradient boosting.
- Use chronological validation for tuning.
- Evaluate once on each held-out supplied test period.
- Save configurations, metrics, predictions, and model metadata.

Deliverables:

- reproducible training command;
- baseline comparison table;
- confusion matrices and precision-recall curves;
- initial model card.

Exit criterion: all models are compared on identical splits and the final test
data has not influenced tuning.

## Phase 4 — Sensor-budget experiments

**Suggested duration:** 3–4 days

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

- ablation results table;
- performance-versus-cost chart;
- recommendation for full-performance and budget deployments.

Exit criterion: the project can quantify what is lost when each sensor is
removed and whether the light sensor is an unacceptable shortcut.

## Phase 5 — Robustness and reliability

**Suggested duration:** 3–5 days

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

- robustness test harness;
- degradation curves;
- failure-mode table;
- reliability recommendation.

Exit criterion: the selected model has documented behavior under every defined
sensor fault.

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

Choose one:

- Streamlit demonstration;
- batch-scoring command;
- small REST API.

Complete:

- final model card;
- reproducibility instructions;
- architecture diagram;
- polished report;
- CI for linting and tests;
- optional container image.

Exit criterion: a new user can install the project, reproduce the headline
result, and run a prediction from documented commands.

## Stretch goals

These should begin only after the core classification study is complete:

- Predict occupancy 5, 15, and 30 minutes ahead.
- Add causal caution around time-of-day and light proxies.
- Detect occupancy transitions as events rather than classifying every row.
- Create an online-learning drift monitor.
- Estimate ventilation-energy savings under different policies.
- Test generalization on a second building or occupancy dataset.

## Suggested headline outputs

For a portfolio-quality final presentation, prioritize:

1. A timeline showing sensor behavior and occupancy transitions.
2. A model comparison table using chronological evaluation.
3. A sensor-ablation heatmap.
4. A performance-versus-cost Pareto chart.
5. Robustness degradation curves.
6. A threshold trade-off tied to operational costs.
7. A short, honest limitations section.
