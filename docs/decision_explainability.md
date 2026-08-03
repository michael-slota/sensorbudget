# Decision threshold and explainability experiment

Phase 6 converts model probabilities into an explicit operating decision and
documents why individual and aggregate predictions behave as they do.

## Leakage controls

The selected Temperature + Light + CO2 logistic model is fixed before this
phase. Five expanding chronological folds regenerate out-of-fold probabilities
for that model only. Error costs and thresholds are evaluated exclusively on
those predictions. Test 1 and Test 2 confirm how the fixed choices generalize;
they never alter the threshold.

## Cost assumptions

Three transparent sensitivity scenarios are configured:

- equal error cost: false occupied and false unoccupied each cost one unit;
- comfort-focused: a false unoccupied decision costs five units;
- energy-focused: a false occupied decision costs five units.

For each scenario, thresholds from 0.01 to 0.99 are compared. Selection
minimizes assumed error cost per 1,000 validation rows, breaking ties with
higher F1 and then proximity to 0.5. Equal error cost is the documented
reference operating scenario, not a claim about real building economics.

## Calibration and explanations

Calibration is diagnosed with Brier score, ten equal-width reliability bins,
and expected calibration error. No calibration transform is fitted because a
reliable comparison would require an additional nested temporal validation
layer.

Global explanations use standardized logistic-regression coefficients. Local
explanations decompose the logit into the intercept plus one contribution from
Temperature, Light, and CO2. Representative rows are chosen independently
within TP, TN, FP, and FN outcomes by proximity to the selected threshold.

Transition analysis reports recall for occupied rows 0–2, 3–5, 6–15, and more
than 15 minutes after each occupancy onset.

## Reproduction

```powershell
python -m sensorbudget.modeling.decision_explainability
```

Generated artifacts are written to `models/decision_explainability/` and are
excluded from version control. The measured interpretation is documented in
`reports/decision_explainability_results.md`.

