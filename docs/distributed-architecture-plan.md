# Migration Plan: Monolithic (Docker Compose) → Distributed (100+ Intersections)

> Status: Draft — pending review
> Scope: runtime architecture, message bus, data platform, deployment, observability
> Out of scope: DQN algorithm changes, model retraining, phase-safety FSM logic

---

## 1. Starting point (as-is)

The current stack (`docker-compose.yml`) is single-host, single-intersection:

| Service | Role | Scaling constraint |
|---|---|---|
| `redis` | Redis Streams as the only bus | 1 node, no partitioning, no retention policy |
| `vision` | 4 cameras → YOLO/TensorRT → `vehicle_counts` | Pinned to 1 GPU, stream list via env var |
| `agent` | `rl.control_runner` → `agent_commands`, `reasoning_logs` | 1 process = 1 intersection, checkpoint mounted read-only |
| `control` | Phase-safety FSM → `phase_states` | No notion of `intersection_id` |
| `api` | FastAPI health/metrics/WebSocket | Telemetry fan-out happens in-process |
| `dashboard` | Streamlit | Reads Redis and the WebSocket directly |
| `mediamtx` + `rtsp-streamers` | RTSP simulation | Demo only |

Streams currently in use (confirmed in code): `vehicle_counts`, `agent_commands`,
`agent_reasoning` / `reasoning_logs`, `phase_states`.

### Three bottlenecks blocking scale

1. **No spatial identity.** Stream names are global (`agent_commands`); there is no
   `intersection_id` in the topic key or the message key. Running two intersections
   against one Redis means two agents stepping on each other.
2. **Unpartitioned bus.** Single-node Redis Streams: throughput ceiling, no
   per-partition replay, no real consumer-group rebalancing when scaling out.
3. **The deployment unit is the whole stack.** Vision (GPU-bound, must sit at the
   edge because of video bandwidth) and Control/RL (CPU-bound, can be centralized)
   are packaged in the same compose file.

---

## 2. Target architecture (to-be)

### 2.1 Principles

- **Edge does the pixels, cloud does the policy.** Video never leaves the
  intersection; only `VehicleCountEvent` (a few hundred bytes) travels upstream.
- **Every message carries `intersection_id`**, and the message key *is*
  `intersection_id`, guaranteeing per-intersection ordering.
- **The safety control loop must survive network loss.** The FSM runs at the edge
  and does not depend on the cloud.
- **Contract-first.** Changing the transport must not change the Pydantic schemas
  in `common/schemas`.

### 2.2 Three tiers

```
┌─ EDGE (per intersection, 1 Jetson/IPC) ─────────────────────────────┐
│  4x camera ─→ vision (TensorRT) ─→ local bus ─→ control (FSM)       │
│                                         │           └→ signal heads  │
│                                         ↓                            │
│                                   edge-gateway (buffer + uplink)     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ MQTT/Kafka over TLS (metadata only)
┌─ REGIONAL / CLOUD ───────────▼───────────────────────────────────────┐
│  Kafka (partitioned by intersection_id)                              │
│    ├→ rl-inference workers  (stateless, N replicas, KEDA autoscale)  │
│    ├→ stream processor (Flink/Faust): aggregation, quality, alerting │
│    ├→ api gateway (FastAPI) → dashboard / mobile / city ops          │
│    └→ sink layer → TimescaleDB (hot) + Parquet/Iceberg on S3 (cold)  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 Decisions to lock in (each becomes an ADR)

| # | Decision | Proposal | Rationale |
|---|---|---|---|
| D1 | Cloud bus | **Kafka (Redpanda if self-hosted)** | partition by `intersection_id`, retention, replay, consumer-group rebalancing — supersedes ADR-0001 |
| D2 | Edge→cloud uplink | **MQTT (EMQX) bridged into Kafka** | tolerates flaky 4G, QoS 1, small payloads; a Kafka client straight from the edge is brittle under NAT and link loss |
| D3 | Edge-local bus | **keep Redis Streams** | already working, sufficient for one intersection, avoids rewriting `streaming/bus` |
| D4 | RL inference location | **cloud, with edge fallback** | a cloud policy can coordinate across intersections; the edge keeps a single-intersection DQN as fallback |
| D5 | Orchestration | **K3s at the edge + K8s (EKS/GKE) in cloud** | one GitOps control plane for both |
| D6 | Hot/cold storage | **TimescaleDB + Iceberg/Parquet on S3** | sub-second dashboard queries; retraining reads from the lakehouse |
| D7 | Model rollout | **MLflow registry + canary by intersection cohort** | stop shipping checkpoints via volume mounts |

---

## 3. Code changes

### 3.1 Contracts (do this first)

`common/schemas/*.py` — add to **every** event:

```python
intersection_id: str        # "hcm.q1.nvc-lelai"
site_id: str                # cluster/region used for routing
schema_version: int = 1
event_ts: datetime          # when the event was produced at the source
ingest_ts: datetime | None  # when the bus received it — measures end-to-end latency
```

Topic naming convention: `traffic.<domain>.v1`, key = `intersection_id`.

| Old (Redis stream) | New (Kafka topic) | Partition key |
|---|---|---|
| `vehicle_counts` | `traffic.counts.v1` | `intersection_id` |
| `agent_commands` | `traffic.commands.v1` | `intersection_id` |
| `phase_states` | `traffic.phase_state.v1` | `intersection_id` |
| `reasoning_logs` / `agent_reasoning` | `traffic.reasoning.v1` | `intersection_id` |

> Note: the two names `reasoning_logs` (`rl/control_runner.py:59`) and
> `agent_reasoning` (`tests/rl_inference.py:106`) are currently inconsistent —
> unify them in this step.

### 3.2 Bus abstraction

`streaming/bus/interface.py` already defines the `MessageBus` Protocol, but
`subscribe()` is still a stub. Required work:

- Finish `subscribe()` in the Protocol with explicit semantics: consumer group,
  at-least-once delivery, explicit ack, `on_error` callback.
- Add `streaming/bus/kafka_client.py` (aiokafka) implementing the same Protocol.
- Add `streaming/bus/mqtt_client.py` for the edge uplink.
- Add `streaming/bus/factory.py` selecting the implementation from
  `BUS_BACKEND=redis|kafka|mqtt`.
- Keep `redis_client.py` — the edge still uses it.

Rule: **no service imports `redis` or `aiokafka` directly**; they import the factory.

### 3.3 Per-intersection services

- `control/service.py:33` currently calls `xadd("phase_states", ...)` — move it to
  the bus abstraction and attach the `intersection_id` from the node's config.
- `rl/control_runner.py` — split into:
  - `rl/inference/worker.py`: stateless; consumes `traffic.counts.v1`, holds
    per-intersection state in memory with a TTL, publishes commands. Scales with
    partition count.
  - `rl/inference/model_cache.py`: loads checkpoints from the MLflow registry by
    `model_uri`, hot-reloads without a pod restart.
- `api/` — drop the in-process fan-out; the WebSocket reads from its own consumer
  group and supports filtering by `intersection_id` or geographic bounding box.

### 3.4 Idempotency and recovery

- Every consumer commits its offset **after** processing (at-least-once), and
  processing must be idempotent on `(intersection_id, event_ts)`.
- Edge gateway: buffer to disk (SQLite/RocksDB) while the uplink is down, replay in
  order on reconnect, and drop anything older than `MAX_STALENESS_S` (default 60s —
  vehicle counts older than that are useless for control).

---

## 4. Data platform

### 4.1 Medallion lakehouse

| Layer | Contents | Format | Retention |
|---|---|---|---|
| Bronze | raw events verbatim from Kafka | Parquet/Iceberg, partitioned `dt/intersection_id` | 90 days |
| Silver | deduplicated, validated, timezone-normalized, phase↔count joined | Iceberg | 1 year |
| Gold | KPIs: avg wait, throughput/h, p95 queue length, vs. fixed-time baseline | Iceberg + TimescaleDB | 3 years |

### 4.2 Hot path

A TimescaleDB hypertable `telemetry` (partitioned by time + `intersection_id`) with
a 1-minute continuous aggregate for the dashboard. The dashboard does **not** read
Kafka directly.

### 4.3 Data quality (gate before Silver)

- Completeness: every intersection must emit ≥ 1 count event per 5s; otherwise
  raise `SENSOR_STALE`.
- Range: `count` ∈ [0, 200] per lane; out of range → quarantine.
- Consistency: every `phase_state` must have a matching `command` within the
  preceding 2s.
- Freshness SLO: p95 end-to-end (camera → gold) < 5 minutes; hot path < 2s.

### 4.4 Orchestration

Airflow (or Dagster) for batch only: hourly Bronze→Silver compaction, daily Gold,
weekly retraining-dataset export. The streaming path does not go through Airflow.

---

## 5. Operations and observability

- **Metrics**: Prometheus + Grafana. Required SLIs: consumer lag per
  topic/partition, p99 decision latency per intersection, FSM command-rejection
  rate (safety vetoes), per-edge-node GPU utilization, uplink availability.
- **Tracing**: OpenTelemetry; the trace id is created in vision and propagated
  through message headers all the way to the dashboard.
- **Logging**: structlog (already in place) → Loki, with `intersection_id` as a
  mandatory field.
- **Alerting**: consumer lag > 30s, sensor stale > 60s, edge offline > 5 min,
  safety veto rate > 5%.
- **Failure modes that must be tested**: uplink loss, Kafka broker down, model
  returning an invalid action, camera frozen but still serving stale frames, edge
  clock skew.

---

## 6. Roadmap

### Phase 0 — Contract foundation (2 weeks) — *no infrastructure changes*
- [ ] Add `intersection_id`, `site_id`, `schema_version`, `event_ts` to `common/schemas`
- [ ] Unify `reasoning_logs` / `agent_reasoning`
- [ ] Finish `MessageBus.subscribe()` in `streaming/bus/interface.py`
- [ ] Refactor `control/service.py` and `rl/control_runner.py` onto the bus abstraction
- [ ] Publish schemas (JSON Schema generated from Pydantic) + CI backward-compatibility check
- **Exit**: the existing compose stack runs unchanged, tests green, `intersection_id` present end-to-end

### Phase 1 — Multi-intersection on one host (2 weeks)
- [ ] Compose override running 3 intersections in parallel on one Redis, isolated by key
- [ ] Intersection selector in the dashboard
- **Exit**: 3 independent intersections, no cross-talk; baseline latency measured

### Phase 2 — Kafka + edge/cloud split (4 weeks)
- [ ] `kafka_client.py`, `mqtt_client.py`, `factory.py`
- [ ] Dual-write to Redis and Kafka, reconcile, then cut over
- [ ] `edge-gateway` with disk buffering
- [ ] New ADRs superseding ADR-0001 and ADR-0003
- **Exit**: 10 simulated intersections; a 10-minute uplink outage loses zero events

### Phase 3 — Data platform (3 weeks)
- [ ] Kafka Connect → S3/Iceberg (Bronze)
- [ ] dbt/Spark jobs for Silver and Gold; TimescaleDB sink
- [ ] Airflow DAGs + data quality gates
- **Exit**: dashboard reads from TimescaleDB, retraining dataset generated automatically

### Phase 4 — Kubernetes + MLOps (4 weeks)
- [ ] Helm charts; K3s at the edge, K8s in cloud; GitOps (ArgoCD/Flux)
- [ ] KEDA autoscaling of inference on consumer lag
- [ ] MLflow registry + canary rollout by cohort
- **Exit**: 100 simulated intersections, p99 decision latency < 200ms

### Phase 5 — Hardening (3 weeks)
- [ ] Chaos testing: broker down, edge offline, model failure
- [ ] mTLS edge↔cloud, per-site credentials, audit logging
- [ ] Runbook + on-call rotation
- **Exit**: 72-hour continuous run at 100 intersections, zero data loss

---

## 7. Load estimate (for sizing)

Assuming 100 intersections × 4 cameras:

| Metric | Value |
|---|---|
| Count events | 100 × 4 cams × 1 Hz ≈ **400 msg/s** |
| Phase state + command | ≈ 100 × 0.2 Hz × 2 = **40 msg/s** |
| Average payload | ~400 B |
| Bus bandwidth | ≈ **0.2 MB/s** (~17 GB/day raw) |
| After Parquet compression (Bronze) | ~2–3 GB/day |
| GPU | 1 edge GPU per intersection (4 streams), **not** centralized |
| Uplink per intersection | < 20 kbps — fits on 4G |

Conclusion: this is **not** a big-data volume problem. The difficulty is the
**number of distributed endpoints, network reliability, and latency budget**.
Investment should therefore go into edge resilience and orchestration, not into a
large Spark cluster.

---

## 8. Risks

| Risk | Level | Mitigation |
|---|---|---|
| Redis→Kafka cutover drops events | High | Dual-write + reconciliation before disabling the Redis path |
| Cloud latency exceeds the control budget | High | Edge FSM/DQN fallback; the cloud optimizes but is never required |
| K8s operating cost overruns | Medium | Start with self-hosted Redpanda + K3s, measure before moving to managed |
| Schema drift between edge and cloud | Medium | Schema registry + CI compatibility check, `schema_version` in the payload |
| Edge node fills its disk with buffer | Low | Cap buffer by capacity and drop by `MAX_STALENESS_S` |

---

## 9. Open questions to settle before Phase 2

1. Self-hosted (Redpanda + K3s) or managed (MSK/Confluent + EKS)? — drives cost and Phase 4.
2. Concrete edge hardware (Jetson Orin Nano/NX?) — determines the TensorRT engine target.
3. Control SLA: how long may an intersection run on fallback when the cloud is unreachable?
4. Legal requirements for traffic data storage (retention, PII if license plates are captured).
