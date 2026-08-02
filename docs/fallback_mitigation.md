# Light-independent fallback experiment

## Question

Can a model that ignores Light recover useful occupancy predictions when the
primary model's Light input is known to be faulty?

This is a mitigation experiment, not a final deployment design. The router is
an **oracle**: it is told when each simulated Light fault is active. Real fault
detection is deliberately left for a later experiment.

## Model selection without held-out leakage

The primary system is the Phase 4 `temperature__light__co2` configuration. Its
eligible fallback candidates are restricted to sensors already installed in
that system:

- Temperature;
- CO2;
- Temperature + CO2.

Light is excluded by definition. The fallback is selected using mean F1 from
the existing five-fold expanding chronological validation on the training
period. Test 1 and Test 2 do not influence this choice.

The selected fallback is Temperature + CO2 with logistic regression. It has a
mean training-validation F1 of 0.679.

## Evaluation strategies

For each held-out period, the experiment records:

- `primary_clean`: the primary model on unchanged data;
- `fallback_only`: the fallback on unchanged data, to expose its performance
  ceiling and clean-data trade-off;
- `primary_under_fault`: the current primary model receiving the simulated
  faulty Light values;
- `oracle_gated_fallback`: the same observations sent to the Light-independent
  fallback because the experiment knows that a fault is active.

The simulated Light conditions are complete loss with median imputation,
stuck-low and stuck-high values, occupied darkness, unoccupied lighting, and
gradual positive or negative drift.

## Reproduction

Run the sensor-budget experiment first so the fitted candidate bundles exist:

```powershell
python -m sensorbudget.modeling.sensor_budget
python -m sensorbudget.robustness.fallback
```

Generated artifacts are written to `models/fallback_mitigation/`:

- `fallback_candidates.csv` records the training-CV selection;
- `fallback_metrics.csv` records held-out metrics for every strategy;
- `metadata.json` records the configuration and routing assumptions.

The generated artifacts are reproducible and excluded from Git. The versioned
analysis is in `notebooks/05_fallback_mitigation.ipynb` and the concise results
are in `reports/fallback_mitigation_results.md`.

## Limitations

- Oracle routing is an upper-bound diagnostic, not a working fault detector.
- The fallback performs differently across the two held-out periods.
- Switching on every small drift is not beneficial; routing requires a fault
  severity or confidence rule.
- This experiment evaluates one primary architecture and makes no final sensor
  recommendation.
