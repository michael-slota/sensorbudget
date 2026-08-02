# Experiment catalog

Experiment IDs make results easier to compare and prevent exploratory work from
silently becoming the final evaluation.

| ID | Question | Feature set or intervention | Priority |
|---|---|---|---:|
| E00 | How hard is the task without learning? | Majority and stratified dummy models | Required |
| E01 | What does an interpretable baseline achieve? | All sensors, logistic regression | Required |
| E02 | Do nonlinear models materially help? | All sensors, tree ensembles | Required |
| E03 | How much does light contribute? | Compare all sensors with no-light | Required |
| E04 | Can a single sensor suffice? | One sensor at a time | Required |
| E05 | What is the best compact sensor set? | All two- and selected three-sensor sets | Required |
| E06 | Does the model merely learn office hours? | With and without calendar features | Recommended |
| E07 | Is humidity ratio redundant? | Remove derived humidity ratio | Recommended |
| E08 | How stable is performance across periods? | Report each supplied test set separately | Required |
| E09 | What happens when data goes missing? | 1%, 5%, 10%, 20%, 40% missingness | Required |
| E10 | What happens when a sensor is stuck? | Freeze one channel at median/high/low | Required |
| E11 | How sensitive is the model to drift? | Gradual offsets by sensor | Recommended |
| E12 | Can fault augmentation improve resilience? | Train with simulated faults | Optional |
| E13 | What threshold minimizes operational cost? | Validation threshold sweep | Required |
| E14 | Are probabilities trustworthy? | Calibration analysis and correction | Recommended |
| E15 | Can occupancy be forecast early? | Shift target by 5/15/30 minutes | Stretch |
| E16 | Can a non-Light fallback recover severe Light failures? | Training-selected fallback with oracle routing | Required |
| E17 | Can Light faults be detected well enough to route safely? | Training-tuned health rules with false alarms and missed faults | Required |

## Required comparison table

Every core experiment should save at least:

- experiment ID and timestamp;
- feature list;
- model and hyperparameters;
- training and validation boundaries;
- selected decision threshold;
- precision, recall, F1, balanced accuracy, PR-AUC, and ROC-AUC;
- training and inference times;
- artifact paths;
- notes explaining anomalies.

## Suggested experiment order

Run `E00`–`E03` first. Complete `E04`, `E05`, and `E08` before adding complex
features. Robustness experiments `E09`–`E12` should use only a small set of
finalist models. Run `E13` after the operational cost assumptions are written.
