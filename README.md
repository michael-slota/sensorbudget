# SensorBudget

SensorBudget is an applied machine-learning project for detecting office
occupancy from environmental sensors while asking a practical question:

> How accurately and reliably can occupancy be detected with fewer, cheaper
> sensors?

The project uses the UCI Occupancy Detection dataset, which contains
time-stamped temperature, humidity, light, CO2, humidity-ratio, and binary
occupancy observations.

## Project goals

1. Build a leakage-safe occupancy-classification baseline.
2. Compare predictive performance across sensor subsets.
3. Quantify the value and risk of relying on the light sensor.
4. Test robustness to sensor noise, missing readings, drift, and failure.
5. Select a model using both predictive quality and operational cost.
6. Package the final pipeline as a reproducible portfolio project.

## Repository layout

```text
sensorbudget/
├── configs/                 # Versioned experiment settings
├── data/
│   ├── external/            # Third-party reference data
│   ├── interim/             # Intermediate transformations
│   ├── processed/           # Model-ready datasets
│   └── raw/                 # Immutable source data
├── docs/
│   ├── decisions/           # Architecture and methodology decisions
│   ├── data_dictionary.md
│   ├── evaluation_plan.md
│   ├── experiment_catalog.md
│   └── roadmap.md
├── models/                  # Serialized models and metadata (not in Git)
├── notebooks/               # Numbered exploratory notebooks
├── references/              # Papers, links, and background notes
├── reports/
│   └── figures/             # Generated charts
├── src/sensorbudget/
│   ├── data/                # Downloading and validation
│   ├── features/            # Feature engineering
│   ├── modeling/            # Training, prediction, and evaluation
│   └── robustness/          # Sensor-fault simulation
└── tests/                   # Automated tests
```

See [docs/roadmap.md](docs/roadmap.md) for the phased delivery plan and
[docs/experiment_catalog.md](docs/experiment_catalog.md) for suggested
experiments. See [docs/model_training.md](docs/model_training.md) for a
step-by-step explanation of the baseline ML pipeline.

## Dataset

- Source: [UCI Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy%2Bdetection)
- DOI: [10.24432/C5X01N](https://doi.org/10.24432/C5X01N)
- License: CC BY 4.0
- Task: binary classification (`Occupancy` is 0 or 1)
- Size: 20,560 observations across three supplied files

Raw files should remain unchanged after download. Their expected location is:

```text
data/raw/
├── datatraining.txt
├── datatest.txt
└── datatest2.txt
```

See [data/README.md](data/README.md) for data-handling rules.

## Proposed modeling approach

The primary evaluation uses the dataset's chronological train/test periods.
Randomly splitting individual rows is not suitable for final reporting because
adjacent minute-level readings are highly correlated.

Initial candidates:

- Dummy classifier as the minimum baseline
- Logistic regression as the interpretable baseline
- Decision tree
- Random forest
- Histogram gradient boosting

The main experiment compares:

- all sensors;
- all sensors except light;
- environmental sensors only;
- each individual sensor;
- selected two- and three-sensor combinations.

Models are evaluated with precision, recall, F1, balanced accuracy, PR-AUC,
ROC-AUC, confusion matrices, calibration, inference cost, and robustness.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Validate the imported source files and build the processed dataset:

```powershell
python -m sensorbudget.data.validate
python -m sensorbudget.data.build
```

The build writes `data/processed/occupancy.csv` and a generated provenance
manifest. Both are reproducible local artifacts and remain outside Git.

Train and evaluate leakage-safe baseline models:

```powershell
python -m sensorbudget.modeling.train
```

This performs expanding chronological validation inside the supplied training
period, selects one all-sensor and one no-Light finalist, and then evaluates
only those finalists on the two held-out periods. Generated models, metrics,
predictions, and metadata are written under `models/baseline/`.

## Working principles

- Preserve temporal ordering during validation.
- Fit every transformation on training data only.
- Keep raw data immutable.
- Version code, configuration, and dataset checksums.
- Separate exploratory notebooks from reusable source code.
- Report operational trade-offs, not accuracy alone.
- Make every reported result reproducible from a committed configuration.

## Current status

The project scaffold, EDA, validated data pipeline, and leakage-safe baseline
comparison are complete. Sensor-subset experiments are the next milestone in
the [roadmap](docs/roadmap.md).
