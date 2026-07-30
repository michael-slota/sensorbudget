# SensorBudget

[![CI](https://github.com/michael-slota/sensorbudget/actions/workflows/ci.yml/badge.svg)](https://github.com/michael-slota/sensorbudget/actions/workflows/ci.yml)

SensorBudget is an applied machine-learning project for detecting office
occupancy from environmental sensors while asking a practical question:

> How accurately and reliably can occupancy be detected with fewer, cheaper
> sensors?

The project uses the UCI Occupancy Detection dataset, which contains
time-stamped temperature, humidity, light, CO2, humidity-ratio, and binary
occupancy observations.

## Headline baseline results

Five classifiers were compared using expanding chronological validation. The
selected all-sensor histogram gradient boosting model was then evaluated on
two untouched, later test periods at the default 0.5 decision threshold.

| Held-out period | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Test 1 | **0.930** | 0.940 | 0.920 | 0.990 |
| Test 2 | **0.883** | 0.830 | 0.943 | 0.991 |

Removing the Light sensor reduced F1 to 0.832 on Test 1 and 0.546 on Test 2,
showing that sensor cost and robustness cannot be judged from aggregate model
accuracy alone. See the
[full baseline report](reports/baseline_results.md) for validation variability,
confusion counts, limitations, and reproduction details.

### Headline visualization

> **Plotly chart placeholder:** the next reporting step will add a
> model-performance-versus-sensor-cost chart here. It will compare the measured
> sensor subsets and highlight the Pareto-efficient deployment choices.

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
|-- configs/                 # Versioned experiment settings
|-- data/
|   |-- external/            # Third-party reference data
|   |-- interim/             # Intermediate transformations
|   |-- processed/           # Model-ready datasets
|   `-- raw/                 # Immutable source data
|-- docs/
|   |-- decisions/           # Architecture and methodology decisions
|   |-- data_dictionary.md
|   |-- evaluation_plan.md
|   |-- experiment_catalog.md
|   `-- roadmap.md
|-- models/                  # Serialized models and metadata (not in Git)
|-- notebooks/               # Numbered exploratory notebooks
|-- references/              # Papers, links, and background notes
|-- reports/
|   `-- figures/             # Generated charts
|-- src/sensorbudget/
|   |-- data/                # Loading, validation, and provenance
|   |-- features/            # Feature engineering
|   |-- modeling/            # Training, prediction, and evaluation
|   `-- robustness/          # Sensor-fault simulation
`-- tests/                   # Automated tests
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
|-- datatraining.txt
|-- datatest.txt
`-- datatest2.txt
```

See [data/README.md](data/README.md) for data-handling rules.

### Dataset attribution and license

The dataset is third-party material and is **not** covered by this project's
MIT software license.

> Candanedo, L. (2016). *Occupancy Detection* [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C5X01N

The dataset is distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0). It may be shared and adapted, including for commercial purposes,
provided appropriate credit is given, the license is linked, and changes are
identified. This project parses the source files, combines their supplied
splits for analysis, validates their contents, and creates derived model
artifacts. The original raw files are not committed to this repository.

## Modeling methodology

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

Evaluate all physical-sensor combinations against the versioned relative-cost
scenario:

```powershell
python -m sensorbudget.modeling.sensor_budget
```

This selects one model per combination using chronological validation, then
reports each selected model on the two later test periods. Generated artifacts
are written under `models/sensor_budget/`.

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
comparison are complete. The sensor-budget phase now evaluates every non-empty
combination of four physical sensors; cost sensitivity, robustness testing,
and the final decision remain. See the [roadmap](docs/roadmap.md) and
[initial sensor-budget results](reports/sensor_budget_results.md). The
interactive analysis is presented in
[`notebooks/03_sensor_budget_analysis.ipynb`](notebooks/03_sensor_budget_analysis.ipynb).

## License

The original software and documentation in this repository are available under
the [MIT License](LICENSE). The UCI dataset remains subject to its separate
CC BY 4.0 license and attribution requirements described above.
