# ADR 0001: Use time-aware evaluation

- Status: Accepted
- Date: 2026-07-23

## Context

The dataset consists of consecutive, minute-level sensor readings. Neighboring
rows are correlated and often describe the same occupancy episode. A random
row split would place near-duplicate conditions in both training and test data.

## Decision

Use the source-provided chronological periods for final evaluation. Use blocked
or expanding-window validation inside the training period for tuning.

## Consequences

- Headline metrics may be lower but better represent deployment on future data.
- Standard shuffled cross-validation utilities are inappropriate by default.
- Transformations and rolling features require careful split-aware fitting.
- Differences between test periods become an important result rather than noise
  to average away.

