# Contributing

## Workflow

1. Create a focused branch.
2. Add or update tests with implementation changes.
3. Run `ruff check .` and `pytest`.
4. Update the relevant documentation and experiment configuration.
5. Keep generated datasets, models, and large figures out of Git.

## Experiment discipline

- Assign an ID from `docs/experiment_catalog.md`.
- Do not inspect held-out test results during feature or threshold selection.
- Record failed and negative experiments when they affect conclusions.
- State assumptions behind synthetic sensor costs and fault simulations.
- Use fixed seeds where supported, while recognizing that reproducibility is
  broader than a seed.

## Commit guidance

Prefer small commits that describe an observable outcome, such as:

```text
feat(data): validate source schema and timestamps
test(features): cover causal rolling-window construction
docs(results): report no-light ablation
```

