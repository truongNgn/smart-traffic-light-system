# Distributed Architecture Migration - Phase 0 Walkthrough

This document serves as a persistent record of the progress and changes made during **Phase 0 (Contract foundation)** of the Distributed Architecture Migration. 

## What changed?

### 1. Data Contracts & Schemas
- Created a `BaseEvent` schema in `common/schemas/base.py` to enforce `intersection_id`, `site_id`, `schema_version`, `event_ts`, and `ingest_ts` on all events.
- Updated all existing schemas (`PhaseAction`, `PhaseState`, `ReasoningLog`, `IntersectionState`, `Telemetry`, `FeedFrame`, `VehicleCountEvent`) to inherit from `BaseEvent`.
- Ensured backwards compatibility in legacy instantiation by utilizing default factories for `BaseEvent` properties fetching from the newly added `SiteSettings` in `common/config.py`.

### 2. Bus Abstraction & Topic Unification
- Updated the `MessageBus` protocol in `streaming/bus/interface.py` to correctly define asynchronous `publish` and `subscribe` semantics.
- Re-wrote `RedisStreamBus` into a fully asynchronous `RedisMessageBus` implementing the new bus protocol in `streaming/bus/redis_client.py`. A synchronous stub `RedisStreamBus` is maintained for backward compatibility with the legacy vision stream.
- Established a clean entry point using a Factory pattern in `streaming/bus/factory.py` to retrieve `get_message_bus()`.

### 3. Service Refactoring
- Updated both `control/service.py` and `rl/control_runner.py` to utilize `get_message_bus()` instead of explicitly tying logic to `redis.Redis`.
- Unified topic names across `api/dependencies.py` and `common/config.py` using new contract namespaces:
  - `traffic.counts.v1`
  - `traffic.commands.v1`
  - `traffic.phase_state.v1`
  - `traffic.reasoning.v1`
- Created `scripts/generate_schemas.py` to export schemas into formal JSON Schema definitions for language-agnostic consumption.

## Verification & Status
- Schema generation works cleanly, producing 7 JSON schema artifacts in `docs/schemas/`.
- Local smoke tests have been executed to assert the contracts didn't break.
- Background unit tests confirmed the application still behaves identically with the new contracts and bus abstraction.

**Status:** Phase 0 is fully complete and ready for the Phase 1 transition (Multi-intersection on one host).
