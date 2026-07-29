# ADR 0002: Treat light as a potentially brittle proxy

- Status: Accepted
- Date: 2026-07-23

## Context

Light is likely to be highly predictive when occupants turn lights on and off.
That relationship may fail if lights are automated, remain on, or daylight
changes. A high-performing model could therefore be operationally fragile.

## Decision

Report every finalist model both with and without light. Include light-only,
stuck-high, and stuck-low experiments. Do not recommend deployment solely from
the best all-sensor score.

## Consequences

- The selected budget model may have lower nominal performance.
- Robustness and transferability become explicit selection criteria.
- The final report can distinguish useful prediction from shortcut learning.

