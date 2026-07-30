# Smart Traffic Light System — Production Implementation Plan

**A Real-Time Streaming, YOLO + Deep Reinforcement Learning Traffic Control System**

> Reference methodology: *Sahal et al. (2023), "Smart Traffic Light Using YOLO Based Camera with Deep Reinforcement Learning Algorithm."* This plan implements the paper's core methodology (80-cell state, yellow/all-red buffers, waiting-time-delta reward, 80→400×5 DQN) with the current codebase's production-safe **2-phase action space**: `EAST_WEST` and `NORTH_SOUTH`. Opposing approaches receive green together, matching a realistic traffic-light program and the checked-in SUMO network.

---

## Table of Contents
1. [Team & Ownership Model](#team--ownership-model)
2. [Target Architecture](#target-architecture)
3. [Repository Layout (Monorepo / Clean Architecture)](#repository-layout)
4. [Cross-Cutting Engineering Standards](#cross-cutting-engineering-standards)
5. [Stage 1 — Foundation](#stage-1)
6. [Stage 2 — Vision Streaming & SUMO State Engine](#stage-2)
7. [Stage 3 — DRL Agent & Control Loop](#stage-3)
8. [Stage 4 — Integration, Training Pipeline & Dashboard](#stage-4)
9. [Stage 5 — Evaluation, Benchmarking & GitHub Showcase](#stage-5)
10. [Definition of Done Matrix](#definition-of-done)

---

## Current Codebase Alignment

This plan is intentionally aligned to the current implementation rather than the earlier 4-single-direction draft:

- **Action space:** 2 discrete phases: `EAST_WEST` and `NORTH_SOUTH`.
- **DQN output:** 2 Q-values, one per phase; the network remains 80 input cells and five 400-unit hidden layers.
- **Compatibility:** legacy `target_direction` payloads are accepted and mapped to the owning phase, but `target_phase` is the canonical control contract.
- **Safety rule:** every phase change still goes through yellow then all-red; unchanged phase commands are idempotent no-ops.
- **Benchmark baseline:** fixed-time alternates the same two axis phases under the same safety buffers as the learned policy.

---

## Team & Ownership Model

| Role | Engineer A — *Vision, Streaming & Backend* | Engineer B — *RL, SUMO & Control Logic* |
|------|--------------------------------------------|------------------------------------------|
| Primary domain | Multi-camera ingestion, YOLO/tracking, message queue, FastAPI, dashboard/WebSockets | SUMO network, TraCI, State/Reward/Action engine, DQN training, benchmarking |
| Owns packages | `vision/`, `streaming/`, `api/`, `dashboard/` | `simulation/`, `rl/`, `control/`, `benchmark/` |
| Shared | `common/` (schemas, config, logging), `docker/`, `.github/`, `docs/` | same |

**Working agreement:** every stage ends with a shared **integration checkpoint** where both engineers merge to `main` behind a green CI run. Contracts between the two halves are defined by **Pydantic schemas in `common/schemas/`** — these are the API boundary and must be agreed *before* parallel work starts each stage.

---

## Target Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                   common/ (contracts)                │
                    │   Pydantic schemas · config · structured logging     │
                    └──────────────────────────────────────────────────────┘
   ENGINEER A                                              ENGINEER B
 ┌───────────────┐   count/speed/pos    ┌──────────────┐   80-cell state     ┌──────────────┐
 │ 4× Video Feed │ ───────────────────► │  Message Bus │ ────────────────►   │  SUMO + TraCI│
 │  Simulator    │                      │ Redis Stream │                     │  Gym Env     │
 └──────┬────────┘                      │  / Kafka     │ ◄──── action ─────  └──────┬───────┘
        │ frames                        └──────┬───────┘   (phase index)            │
 ┌──────▼────────┐                             │                             ┌──────▼───────┐
 │ YOLOv8/v11 +  │                      ┌──────▼───────┐                     │ DQN Agent    │
 │ ByteTrack     │                      │  FastAPI +   │                     │ (PyTorch)    │
 │ Centroid Cnt  │                      │  WebSockets  │◄───reasoning logs───│ + Reward Eng │
 └───────────────┘                      └──────┬───────┘                     └──────┬───────┘
                                               │                                    │
                                        ┌──────▼────────┐                    ┌──────▼────────┐
                                        │  Dashboard    │                    │Control Service│
                                        │ (4 feeds +    │                    │ Yellow/All-Red│
                                        │  SUMO view)   │                    │ safety buffers│
                                        └───────────────┘                    └───────────────┘
```

**Data contracts (the two seams that must never drift):**
- `VehicleCountEvent` — Vision → Bus → everyone (per feed: counts by class, speeds, centroids).
- `IntersectionState` — SUMO → Agent (80-cell grid vector + metadata).
- `PhaseAction` — Agent → Control Service (action ∈ {EAST_WEST, NORTH_SOUTH} + safety-phase intent).

---

## Repository Layout

```
smart-traffic-system/
├── common/                     # SHARED — the contract layer (owned jointly)
│   ├── schemas/                # Pydantic models: events, state, action, telemetry
│   ├── config/                 # pydantic-settings; env-driven, 12-factor
│   ├── logging/                # structlog JSON logger factory
│   └── constants.py            # 80 cells, 2 phases, 2s yellow, 2s all-red, 100m, 4 classes
├── vision/                     # ENGINEER A
│   ├── ingestion/              # multi-feed video simulator (loop mp4 → RTSP-like)
│   ├── detection/              # YOLOv8/v11 wrapper (ultralytics)
│   ├── tracking/               # ByteTrack + centroid tracker
│   └── producer/               # publishes VehicleCountEvent to bus
├── streaming/                  # ENGINEER A
│   ├── bus/                    # Redis Streams / Kafka abstraction (Protocol interface)
│   └── pubsub/                 # fan-out to WebSocket clients
├── api/                        # ENGINEER A — FastAPI app (WebSockets, REST health/metrics)
├── dashboard/                  # ENGINEER A — Streamlit (v1) → React (stretch)
├── simulation/                 # ENGINEER B
│   ├── net/                    # .net.xml, .rou.xml, .sumocfg (Netedit output)
│   ├── traci_wrapper/          # connection lifecycle, step loop, safe teardown
│   └── state/                  # TraCI vehicle states → 80-cell grid
├── rl/                         # ENGINEER B
│   ├── env/                    # Gymnasium env wrapping SUMO+TraCI
│   ├── agent/                  # DQN (PyTorch), replay buffer, target net
│   ├── reward/                 # waiting-time-delta engine
│   └── train/                  # training loop, MLflow/W&B, checkpoints
├── control/                    # ENGINEER B — phase-switch executor (yellow/all-red FSM)
├── benchmark/                  # ENGINEER B — fixed-time vs RL, metric collectors
├── tests/                      # pytest; unit + integration; mirrors package tree
├── docker/                     # per-service Dockerfiles
├── docs/                       # architecture diagrams, ADRs, benchmark reports
├── .github/workflows/          # CI: lint, type-check, test, build
├── docker-compose.yml          # full multi-container stack
├── pyproject.toml              # ruff + mypy + pytest config, deps (uv/poetry)
└── README.md                   # the showcase piece
```

---

## Cross-Cutting Engineering Standards

Apply from commit #1 — these are graded continuously, not bolted on at Stage 5.

- **Language/tooling:** Python 3.11+, `uv` or `poetry` for deps, `ruff` (lint+format), `mypy --strict` on `common/`, `rl/`, `simulation/`.
- **Type hints:** mandatory on all public functions; Pydantic v2 for all data crossing a boundary.
- **Clean Architecture:** dependencies point inward — `vision`/`rl` depend on `common`, never the reverse. Bus and detector accessed through `Protocol` interfaces so they're swappable and mockable.
- **Testing:** `pytest` + `pytest-asyncio`; ≥70% coverage on core logic (state encoder, reward, FSM, DQN forward-pass shapes). Use `pytest` fixtures + fakes for SUMO/bus in unit tests.
- **Structured logging:** `structlog` JSON logs with correlation IDs per simulation episode and per frame batch.
- **Config:** `pydantic-settings`, all knobs via env / `.env`; no magic numbers in code — they live in `common/constants.py`.
- **CI/CD (`.github/workflows/ci.yml`):** on every PR run `ruff check`, `mypy`, `pytest`, and `docker build`. Block merge on red.
- **Containers:** one Dockerfile per service, multi-stage builds, non-root user, pinned base images.
- **Git hygiene:** trunk-based with short-lived feature branches, Conventional Commits, PR template, CODEOWNERS mapping `vision/`→A and `rl/`→B.

---

<a id="stage-1"></a>
## Stage 1 — System Architecture, Environment Setup & Data Pipeline Foundation

**Stage Goal:** A running skeleton monorepo where each half boots in Docker, the contract schemas exist, and each engineer has a "hello world" of their vertical (a detection frame published; a SUMO net stepping via TraCI).

**Deliverables:** monorepo scaffold, `common/schemas`, CI pipeline green, two Dockerfiles building, one demo video looping through YOLO, one SUMO intersection stepping.

### Engineer A
1. Scaffold `common/` with **Pydantic schemas** `VehicleCountEvent`, `FeedFrame`, `Telemetry`; wire `structlog` + `pydantic-settings`.
2. **Video stream simulator** in `vision/ingestion/`: read local `.mp4` files (one per direction), loop them, emit timestamped frames at a fixed FPS — a stand-in for 4 RTSP cameras.
3. **Dockerized YOLO + tracking**: `vision/detection` (ultralytics YOLOv8n/v11n) + `vision/tracking` (ByteTrack) + centroid computation (midpoint of bbox). Draw and save annotated frames as a smoke test.
4. **Bus producer**: `streaming/bus/` Redis Streams client behind a `MessageBus` Protocol; publish a hardcoded `VehicleCountEvent` end-to-end.

### Engineer B
1. **SUMO network via Netedit**: build a 4-way intersection, multi-lane approaches, ~100m segments per direction, explicit left/right-hand rule for the target locale; export `.net.xml`, `.rou.xml`, `.sumocfg`.
2. **Route/demand generation**: `randomTrips.py` or handwritten `.rou.xml` producing the 4 vehicle classes (car, motorcycle, bus, truck) with realistic ratios.
3. **TraCI wrapper**: `simulation/traci_wrapper/` — connection lifecycle, `step()`, deterministic seeding, guaranteed `close()` (context manager), headless (`sumo`) + GUI (`sumo-gui`) toggle.
4. Smoke test: step the sim N ticks, assert vehicles spawn and the traffic-light program is queryable via TraCI.

### Shared Deliverables & Integration Checkpoint
- ✅ `common/schemas` reviewed and merged first — this unblocks both sides.
- ✅ `docker-compose up` brings Redis + a placeholder vision container + a SUMO container.
- ✅ CI green: lint, type-check, tests, both docker builds.
- **Checkpoint demo:** A publishes a real (if crude) count event to Redis; B shows SUMO stepping headless in a container.

### Production Readiness
`pyproject.toml` with ruff/mypy/pytest; CI workflow file; PR template + CODEOWNERS; first ADR in `docs/adr/0001-message-bus-choice.md` (Redis Streams vs Kafka — recommend **Redis Streams** for this scale, Kafka as documented future path).

---

<a id="stage-2"></a>
## Stage 2 — Core Vision Streaming & SUMO Integration Engine

**Stage Goal:** Both verticals produce their real data structures continuously: A streams live per-lane counts/speeds/positions over WebSockets; B converts raw SUMO state into the paper's **80-cell grid**.

**Deliverables:** live WebSocket telemetry feed; validated 80-cell state encoder with unit tests proving the cell mapping.

### Engineer A
1. **Real-time pipeline**: FastAPI app (`api/`) that consumes the bus and re-broadcasts via **WebSockets**; add Redis Pub/Sub fan-out so multiple dashboard clients can subscribe.
2. Enrich `VehicleCountEvent` with per-class counts, mean speed, and centroid positions per lane; publish at a steady cadence.
3. Per-lane counting logic: virtual counting lines / ROI polygons per approach; de-duplicate via track IDs to avoid double counts.
4. REST endpoints: `/health`, `/metrics` (Prometheus-style), `/feeds` metadata. Backpressure handling for slow WS clients.

### Engineer B
1. **State extractor** (`simulation/state/`): implement the paper's discretization — each 100m approach split into cells (the paper's 80-cell total across the intersection), each cell a binary/scaled occupancy from `traci.lane`/`traci.vehicle` positions.
2. Produce a fixed-length **80-dim vector** (`IntersectionState` schema) each control step; document the exact cell indexing (which cells map to which lane/offset) in `docs/state_encoding.md`.
3. Waiting-time primitives: expose accumulated waiting time per vehicle (speed < 0.1 m/s) — the substrate for the Stage-3 reward.
4. Golden tests: hand-craft SUMO scenarios (e.g., one car at a known offset) and assert exactly which cell lights up.

### Shared Deliverables & Integration Checkpoint
- ✅ `IntersectionState` and enriched `VehicleCountEvent` finalized in `common/schemas`.
- ✅ A optional bridge: vision counts can seed/validate SUMO demand (documented as a "digital-twin calibration" hook).
- **Checkpoint demo:** open the WS stream in a browser tab and watch counts update; run the encoder over a live SUMO episode and print the 80-vector each step.

### Production Readiness
Async tests with `pytest-asyncio` + `httpx` for the API; WebSocket contract test; property-based test (`hypothesis`) that the encoder always returns an 80-vector in `[0,1]`. Add `/metrics` scraping to compose.

---

<a id="stage-3"></a>
## Stage 3 — Deep Reinforcement Learning Agent & Control Loop Development

**Stage Goal:** A trainable DQN that consumes the 80-cell state and emits one of 2 axis phases, plus a safety-correct control executor that inserts the mandatory yellow + all-red buffers.

**Deliverables:** PyTorch DQN (80→400×5→2), Gymnasium env, reward engine, and a phase-switch FSM proven safe by tests.

### Engineer A
1. **Decision-handling / control-facing service** (`control/` interface side + API): receive `PhaseAction` from the agent over the bus and drive the executor; expose current phase + reasoning to the dashboard.
2. **Phase-switch safety executor**: finite-state machine enforcing — on any phase change, insert **2s yellow → 2s all-red** before the new green; no-op fast path when the chosen action equals the current green.
3. Emit **AI reasoning logs** (chosen action, Q-values, buffer state) as structured events for the dashboard.
4. Idempotency + safety: reject illegal transitions; watchdog that fails safe to all-red if the agent stalls.

### Engineer B
1. **DQN in PyTorch** (`rl/agent/`): input 80, **5 hidden layers of 400 neurons** (ReLU), output 2 Q-values; target network + experience replay + ε-greedy; Huber loss, Adam.
2. **Gymnasium env** (`rl/env/`) wrapping SUMO+TraCI: `reset()` starts an episode, `step(action)` applies the action *through the safety buffers*, advances SUMO, returns `(state80, reward, terminated, truncated, info)`.
3. **Reward engine** (`rl/reward/`): `r_t = t_{t-1}^T − t_t^T` where `t^T` is accumulated total waiting time of vehicles with speed < 0.1 m/s. Unit-test with scripted waiting-time sequences.
4. Action space = {`EAST_WEST`, `NORTH_SOUTH`}; ensure env applies the same yellow/all-red buffers as the production executor (single source of truth in `common/constants.py`).

### Shared Deliverables & Integration Checkpoint
- ✅ **One canonical phase-transition spec** shared by the RL env (B) and the production executor (A) — critical to avoid sim-vs-real drift.
- ✅ `PhaseAction` schema finalized; reasoning-log schema added.
- **Checkpoint demo:** run a short training session; watch reward trend upward over episodes; verify in `sumo-gui` that every direction change shows yellow→all-red.

### Production Readiness
Deterministic seeding for reproducibility; DQN shape tests (forward pass on batch → `[B,2]`); FSM property tests enumerating all from→to phase pairs; reward function tested against hand-computed deltas. ADR `0002-reward-and-safety-buffers.md`.

---

<a id="stage-4"></a>
## Stage 4 — System Integration, Training Pipeline & Real-Time Dashboard

**Stage Goal:** Everything runs together: a full training pipeline with experiment tracking and baselines, and a live dashboard showing the whole system reasoning in real time.

**Deliverables:** reproducible training pipeline with MLflow/W&B, fixed-time baseline, and a monitoring dashboard.

### Engineer A
1. **Monitoring dashboard** (`dashboard/`): **Streamlit v1** (fast) showing — 4 annotated camera feeds, live **SUMO view** (rendered frames or GUI screenshot stream), per-lane queue metrics, current phase + countdown, and the **AI reasoning log** (action + Q-values).
2. Wire dashboard to the WebSocket/Pub-Sub layer; add historical charts (queue length, waiting time) via a small time-series buffer.
3. (Stretch) React + WebSocket front-end as `dashboard-web/` once Streamlit version is signed off.
4. Deployment polish: dashboard container in compose, env-configurable endpoints.

### Engineer B
1. **End-to-end training pipeline** (`rl/train/`): episode loop, replay warm-up, target-net sync cadence, ε decay schedule, **checkpoint saving** (best + periodic), resume-from-checkpoint.
2. **Experiment tracking**: MLflow (self-hosted, composes cleanly) or W&B — log reward, epsilon, loss, mean queue length, mean waiting time per episode; save config as artifact.
3. **Baseline controllers** for comparison: **fixed-time** (paper's reference) and optionally max-pressure/actuated; run under identical demand + seeds.
4. Evaluation harness: load a checkpoint, run deterministic eval episodes, dump metrics to `benchmark/results/`.

### Shared Deliverables & Integration Checkpoint
- ✅ Trained agent's actions flow through the **real control service and executor**, not just the env — proving the sim-to-service contract holds.
- ✅ Dashboard renders live during a training/eval run.
- **Checkpoint demo:** side-by-side dashboard panels of Fixed-time vs RL on the same demand; RL shows lower queues.

### Production Readiness
Training config fully env/file-driven; runs reproducible from a logged seed + config; MLflow container in compose; dashboard has a health check. Coverage gate enforced in CI.

---

<a id="stage-5"></a>
## Stage 5 — Evaluation, Benchmarking, Production Optimization & GitHub Showcase

**Stage Goal:** Prove the system works, package it for one-command startup, and turn the repo into a recruiter-grade portfolio piece.

**Deliverables:** benchmark report with charts, full `docker-compose` stack, demo GIFs, and a polished `README.md`.

### Both Engineers
1. **End-to-end integration tests**: a smoke test that boots the compose stack, runs a short episode through vision→bus→agent→control→dashboard, and asserts telemetry flows and no crashes.
2. **Docker-Compose multi-container deployment**: Redis, vision, api, dashboard, control, sumo/rl-runner — one `docker-compose up` brings the whole system. Health checks + restart policies + resource limits.
3. **Benchmarking** (`benchmark/`): across multiple demand levels and seeds, report **queue-length reduction**, **average/total waiting-time reduction**, and **throughput (vehicles cleared)** — RL vs fixed-time. Generate matplotlib charts (see `dataviz` guidance) into `docs/benchmarks/`.
4. **Performance optimization**: profile the vision loop (batch inference, half-precision, frame skipping) and the training loop (vectorized replay, GPU); document FPS and steps/sec achieved.

### GitHub Showcase Artifacts
- **`README.md`**: hero architecture diagram, one-paragraph pitch, feature list, **demo GIFs** (dashboard live, sumo-gui with RL control, benchmark chart), quickstart (`docker-compose up`), and a "Results" section with the headline numbers.
- **`docs/`**: architecture diagram (Mermaid), ADRs, state-encoding spec, benchmark report, and a short "How the DRL works" explainer tying back to Sahal et al. (2023).
- **Repo polish**: badges (CI, coverage, license, Python version), `LICENSE`, `CONTRIBUTING.md`, issue/PR templates, tagged `v1.0.0` release with attached benchmark PDF.

### Production Readiness
Full stack reproducible from clean clone; CI runs the integration smoke test; documented hardware requirements; secrets via env only; teardown script. Final ADR `0003-deployment-topology.md`.

---

<a id="definition-of-done"></a>
## Definition of Done Matrix

| Capability | Stage | Owner | Proof |
|-----------|-------|-------|-------|
| Monorepo + CI green | 1 | Both | GitHub Actions passing |
| YOLO + ByteTrack counts published | 1–2 | A | WS stream shows live counts |
| SUMO 4-way + TraCI stepping | 1 | B | Headless episode runs |
| 80-cell state encoder | 2 | B | Golden cell-mapping tests |
| Live WebSocket telemetry | 2 | A | Browser subscribes, updates |
| DQN 80→400×5→2 | 3 | B | Shape + training-signal tests |
| Yellow/All-Red safety FSM | 3 | A | All from→to transition tests + sumo-gui |
| Reward = waiting-time delta | 3 | B | Hand-computed reward tests |
| Training pipeline + tracking | 4 | B | MLflow run with rising reward |
| Live dashboard (4 feeds + SUMO) | 4 | A | Dashboard renders during run |
| Fixed-time vs RL benchmark | 4–5 | B | Comparison charts |
| One-command compose deploy | 5 | Both | `docker-compose up` smoke test |
| Showcase README + GIFs | 5 | A | Rendered on GitHub |

---

### Recommended Sequencing Notes
- **Freeze `common/schemas` at the start of every stage.** Most integration pain in dual-track projects comes from contract drift; the schema is the treaty.
- **Single source of truth for phase timing.** The RL env (B) and the production executor (A) must import the *same* yellow/all-red constants, or the trained policy will be unsafe in deployment.
- **Redis Streams over Kafka** for a 4-intersection portfolio system — lower ops burden, same conceptual demonstration; note Kafka as the documented scale-out path.
- **Streamlit first, React as stretch.** Ship a working dashboard early; upgrade the front-end only after the pipeline is proven.
