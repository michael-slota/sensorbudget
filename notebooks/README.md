# Notebooks

Use notebooks for exploration and communication, not as the only implementation
of a pipeline.

Notebook sequence:

1. `01_data_audit.ipynb` — complete data audit and time-aware EDA
2. `02_baseline_models.ipynb` — Plotly review of packaged baseline artifacts
3. `03_sensor_budget_analysis.ipynb` — sensor ablation, validation Pareto
   frontier, cost scenarios, and held-out stability
4. `04_robustness.ipynb` — sensor-failure, missingness, noise, and drift
   analysis
5. `05_fallback_mitigation.ipynb` — Light-independent fallback and oracle
   routing analysis
6. `06_explainability.ipynb` — planned decision and explanation layer

Move reusable parsing, feature, plotting, and evaluation functions into
`src/sensorbudget/`. Clear large cell outputs before committing unless the
notebook is intended as a final report.

The notebooks present results produced by reusable code under
`src/sensorbudget/`. Run the corresponding package commands before opening a
notebook when its generated artifacts are absent.
