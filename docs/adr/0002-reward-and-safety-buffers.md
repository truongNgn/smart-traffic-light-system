# ADR 0002: Waiting-Time Reward and Shared Safety Buffers

## Status

Accepted

## Context

The RL environment and production control service must agree on traffic-light
timing. If training skips yellow/all-red buffers that production enforces, the
learned policy observes a different world from the deployed controller.

## Decision

Keep phase timing constants in `common/constants.py` and import them from both
`rl/env/` and `control/`.

The reward remains the waiting-time delta:

```text
r_t = total_waiting_time_(t-1) - total_waiting_time_t
```

The action space is the current 2-phase contract:

- `EAST_WEST`
- `NORTH_SOUTH`

Every phase change inserts yellow then all-red before granting the new green.
An unchanged phase command is an idempotent no-op.

## Consequences

The trained policy optimizes under the same safety timing that the runtime
enforces. The 2-phase action space is smaller than the original 4-direction
draft, but it better matches a realistic two-axis signal program for the
checked-in SUMO network.
