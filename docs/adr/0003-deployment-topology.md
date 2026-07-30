# ADR 0003: Docker Compose Deployment Topology

## Status

Accepted

## Context

The project needs a one-command local demo that proves the model, control
service, API, dashboard, vision smoke producer, and Redis bus can run together.

## Decision

Use Docker Compose as the production-like local topology:

- `redis` for streams
- `api` for health, metrics, and WebSocket telemetry
- `control` for phase safety execution
- `dashboard` for live observability
- `agent` for trained DQN inference against SUMO
- `vision` for YOLO smoke telemetry

## Consequences

Compose is easy to run from a clean clone and mirrors the service boundaries
without adding orchestration overhead. Kubernetes or managed stream services can
be introduced later without changing the internal contracts.
