# Results and reports

## Start here

Read the [model card](model_card.md) for the consolidated candidate, results,
known failure modes, and evidence limits. For a shorter visual review, use the
[published dashboard suite](https://michael-slota.github.io/sensorbudget/).

## Main evidence path

| Question | Report | Main result |
|---|---|---|
| Is the source data suitable for time-aware modeling? | [Data quality and EDA](data_quality.md) | 20,560 validated rows; Light is both predictive and a proxy risk. |
| Which initial model performs best? | [Baseline comparison](baseline_results.md) | Histogram boosting leads the initial all-sensor comparison, with period-dependent held-out F1. |
| Which sensor subsets preserve performance? | [Sensor-budget analysis](sensor_budget_results.md) | Three Light-containing configurations remain Pareto-efficient under five illustrative cost scenarios. |
| What happens when sensors become unreliable? | [Robustness analysis](robustness_results.md) | Complete Light loss and occupied darkness reduce F1 to approximately zero. |
| Can the system make a better operating decision? | [Decision and explainability](decision_explainability_results.md) | A validation-selected threshold does not remain advantageous on held-out periods. |

## Mitigation deep dives

- [Oracle fallback](fallback_mitigation_results.md): establishes the potential
  value and performance ceiling of a Light-independent fallback.
- [Fault detection and routing](fault_detection_results.md): replaces perfect
  fault knowledge with causal rules and measures false alarms and missed faults.
- [Fault-aware training](fault_aware_results.md): compares augmentation and a
  missingness indicator against detector routing.

Every report links to its experiment configuration, notebook, and source-data
checksums. Generated row-level artifacts remain outside version control and can
be reproduced with the commands in the repository README.
