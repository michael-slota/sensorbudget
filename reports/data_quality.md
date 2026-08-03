# Occupancy dataset quality and EDA report

> **Project status:** This report records the Phase 1–2 source audit and EDA.
> The data pipeline and all later modeling phases are now complete. Continue
> with the [baseline report](baseline_results.md), or see the consolidated
> [model card](model_card.md).

**Traceability:** The analysis is implemented in
[`notebooks/01_data_audit.ipynb`](../notebooks/01_data_audit.ipynb), and source
identity is fixed by [`data/source_checksums.json`](../data/source_checksums.json).
The Git commit containing this report identifies the corresponding code and
documentation revision.

## Scope

This report summarizes the initial audit of the UCI Occupancy Detection files:

| Supplied split | File | Rows | Period |
|---|---|---:|---|
| `train` | `datatraining.txt` | 8,143 | 2015-02-04 to 2015-02-10 |
| `test_1` | `datatest.txt` | 2,665 | 2015-02-02 to 2015-02-04 |
| `test_2` | `datatest2.txt` | 9,752 | 2015-02-11 to 2015-02-18 |

Total observations: 20,560.

The source header omits the row-ID field that appears first in every data row.
The notebook therefore supplies an explicit eight-column schema and names that
field `source_row_id`.

## Data-quality results

- All files match the expected schema.
- No missing cells were found.
- No duplicate rows or duplicate timestamps were found.
- Timestamps are monotonically increasing within each source file.
- Sampling intervals range from 59 to 61 seconds, with a median of 60 seconds.
- No within-file gaps exceed 90 seconds.
- The target contains only `0` (unoccupied) and `1` (occupied).

The files are suitable for analysis without imputation or duplicate resolution.
The supplied split identity must remain attached to every row.

## Target balance

| Split | Unoccupied | Occupied | Occupied rate |
|---|---:|---:|---:|
| `train` | 6,414 | 1,729 | 21.2% |
| `test_1` | 1,693 | 972 | 36.5% |
| `test_2` | 7,703 | 2,049 | 21.0% |

The class prevalence in `test_1` differs notably from training and `test_2`.
Accuracy alone is therefore unsuitable. Precision, recall, F1, balanced
accuracy, PR-AUC, and confusion matrices should be reported per test period.

## Schedule patterns

Observed occupancy is strongly schedule-dependent:

- No occupied observations occur during hours 00:00–06:59 or 19:00–23:59.
- Hourly occupancy is highest around 09:00 and 16:00–17:59.
- No occupied observations occur on the Saturdays and Sundays represented.

The collection period is short, so weekday estimates are based on few calendar
dates. Hour and weekday may be predictive features, but they may encode this
particular office routine. Models should be compared with and without calendar
features.

## Sensor relationships

Pearson correlations with occupancy across all periods are approximately:

| Sensor | Correlation |
|---|---:|
| Light | 0.915 |
| Temperature | 0.556 |
| CO2 | 0.502 |
| HumidityRatio | 0.257 |
| Humidity | 0.046 |

These are descriptive associations rather than causal effects or guarantees of
future performance.

`HumidityRatio` is derived from Temperature and Humidity rather than measured by
an additional physical sensor. It should be retained in the initial baseline
and removed in a later ablation to test whether it adds useful representation.

## Light as a proxy

An exact-zero Light reading was treated as dark for this audit.

| Split | Occupied while dark | Unoccupied while lit |
|---|---:|---:|
| `train` | 0 of 1,729 | 1,254 of 6,414 |
| `test_1` | 0 of 972 | 78 of 1,693 |
| `test_2` | 1 of 2,049 | 1,707 of 7,703 |

Occupancy is almost never observed in complete darkness, making Light an
extremely strong signal. The reverse rule is unreliable: many unoccupied rows
still have positive Light readings. Lighting policies, daylight, automation,
or sensor faults could change this relationship, so every finalist must also be
evaluated without Light.

## Occupancy episodes

An episode is a consecutive run in one occupancy state. Durations include both
the first and final sampled minute.

Typical occupied episodes last tens of minutes:

| Split | Occupied episodes | Median duration | Maximum duration |
|---|---:|---:|---:|
| `train` | 21 | 24 min | 270 min |
| `test_1` | 14 | 35 min | 274 min |
| `test_2` | 25 | 35 min | 284 min |

Unoccupied episodes are strongly skewed by long overnight and weekend periods.
Long easy runs can dominate row-level metrics, so evaluation should include
errors near arrivals and departures.

## Distribution shift

Standardized mean difference (SMD) compares each test mean with the training
mean in pooled standard-deviation units:

| Split | Temperature | Humidity | Light | CO2 | HumidityRatio |
|---|---:|---:|---:|---:|---:|
| `test_1` | 0.80 | -0.09 | 0.33 | 0.37 | 0.22 |
| `test_2` | 0.38 | 0.87 | 0.02 | 0.48 | 1.02 |

The most prominent shifts are Temperature in `test_1`, and Humidity and
HumidityRatio in `test_2`. Both held-out periods should be reported separately.
Randomly shuffling adjacent rows would obscure this temporal shift and create
an optimistic evaluation.

## Modeling implications

1. Preserve the source-provided periods and tune only within `train`.
2. Use time-aware validation rather than random row-level splitting.
3. Report class-sensitive metrics separately for `test_1` and `test_2`.
4. Establish all-sensor and no-Light baselines.
5. Compare models with and without calendar features.
6. Compare models with and without `HumidityRatio`.
7. Add transition-focused error analysis alongside row-level metrics.
8. Treat the dataset as a single-room, short-period study; broader building
   generalization is not established.

## Reproduction

The calculations and interactive Plotly figures are in
[`notebooks/01_data_audit.ipynb`](../notebooks/01_data_audit.ipynb). Its schema,
validation rules, and checksum verification have since been implemented in
[`src/sensorbudget/data/`](../src/sensorbudget/data/).
