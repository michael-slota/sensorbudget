# Notebooks

Use notebooks for exploration and communication, not as the only implementation
of a pipeline.

Planned sequence:

1. `01_data_audit.ipynb` — complete data audit and time-aware EDA
2. `02_baseline_models.ipynb` — Plotly review of packaged baseline artifacts
3. `03_sensor_ablation.ipynb`
4. `04_robustness.ipynb`
5. `05_explainability_and_cost.ipynb`

Move reusable parsing, feature, plotting, and evaluation functions into
`src/sensorbudget/`. Clear large cell outputs before committing unless the
notebook is intended as a final report.

Run `01_data_audit.ipynb` before implementing the formal data pipeline. Its
observations should define the first validation rules and data-quality report.
