# Run Guide

This guide describes the current integrated runtime after merging the vision/backend work with the SUMO/RL/control work.

The working live flow is:

```text
SUMO env -> DQN checkpoint runner -> Redis agent_commands -> control service
                                \-> Redis reasoning_logs -> API -> dashboard
vision producer/smoke test ------> Redis vehicle_counts -> API -> dashboard
control service -----------------> Redis phase_states -> API -> dashboard
```

## Prerequisites

- Docker Desktop
- A trained checkpoint in `checkpoints/`
- Python 3.11+, `uv`, and local SUMO only if you want to run services outside Docker

Recommended checkpoint:

```text
checkpoints/dqn_ew_repair_best.pt
```

## 1. Start the Full Docker Stack

From the repository root:

```bash
docker compose up --build
```

This builds and starts:

- `redis`
- `api`
- `control`
- `dashboard`
- `agent`
- `vision`

Open the dashboard:

```text
http://localhost:8502
```

Open the API health endpoint:

```text
http://localhost:8000/health
```

Stop everything:

```bash
docker compose down
```

Check service status:

```bash
docker compose ps
```

Docker publishes the dashboard on host port `8502` while Streamlit still runs on container port `8501`. This avoids collisions with a local Streamlit dev session. Override host ports when needed:

```bash
REDIS_HOST_PORT=6380 API_HOST_PORT=8001 DASHBOARD_HOST_PORT=8503 docker compose up --build
```

## 2. Docker Debug Commands

Read stream lengths:

```bash
docker exec smart-traffic-system-redis-1 redis-cli XLEN vehicle_counts
docker exec smart-traffic-system-redis-1 redis-cli XLEN agent_commands
docker exec smart-traffic-system-redis-1 redis-cli XLEN phase_states
docker exec smart-traffic-system-redis-1 redis-cli XLEN reasoning_logs
```

Follow logs:

```bash
docker compose logs -f api control dashboard agent vision
```

Rebuild one service:

```bash
docker compose build api
docker compose up -d api
```

## 3. Local Development Mode

Use this mode when you want to run services manually with `uv`.

Install Python dependencies:

```bash
uv sync
```

Start Redis:

```bash
docker compose up -d redis
```

### Terminal 1: API Gateway

Open a new terminal:

```bash
uv run python -m uvicorn api.main:app --reload
```

The API consumes Redis streams and broadcasts events through:

```text
ws://localhost:8000/ws/telemetry
```

Health check:

```bash
curl http://localhost:8000/health
```

### Terminal 2: Control Service

Open a new terminal:

```bash
uv run python -m control.service
```

The control service listens to:

```text
agent_commands
```

It publishes:

```text
phase_states
```

The service enforces yellow/all-red safety transitions and fails safe to all-red if commands stop arriving.

### Terminal 3: Dashboard

Open a new terminal:

```bash
uv run streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

The dashboard displays:

- live lane/count telemetry
- current phase state
- active green/yellow/all-red status
- DQN reasoning logs and Q-values

### Terminal 4: Trained DQN Agent

Open a new terminal:

```bash
uv run python -m rl.control_runner ^
  --checkpoint checkpoints/dqn_ew_repair_best.pt ^
  --sumocfg simulation/net/intersection.sumocfg ^
  --episode-duration 300 ^
  --backend traci
```

PowerShell one-line version:

```powershell
uv run python -m rl.control_runner --checkpoint checkpoints/dqn_ew_repair_best.pt --sumocfg simulation/net/intersection.sumocfg --episode-duration 300 --backend traci
```

The runner loads the checkpoint, starts the SUMO environment, selects phases with the DQN, and publishes:

```text
agent_commands
reasoning_logs
```

For a fast smoke test:

```bash
uv run python -m rl.control_runner ^
  --checkpoint checkpoints/dqn_ew_repair_best.pt ^
  --sumocfg simulation/net/intersection.sumocfg ^
  --episode-duration 60 ^
  --backend traci ^
  --decision-interval 0
```

Use `--decision-interval 0` only for quick tests. For a live dashboard demo, keep the default interval so the control FSM has time to show transitions.

## 4. Run Only the Vision Smoke Test

The Docker vision service currently runs a short smoke test and exits when complete.

Build:

```bash
docker compose up --build redis vision
```

Inspect logs:

```bash
docker compose logs -f vision
```

Check vehicle event count:

```bash
docker exec smart-traffic-system-redis-1 redis-cli XLEN vehicle_counts
```

## 5. Optional Mock Agent

Use this only to test the control service without running SUMO/model:

```bash
uv run python -m tests.mock_agent
```

The mock sends two phase commands:

- `EAST_WEST`
- `NORTH_SOUTH`

## 6. Useful Redis Debug Commands

Read the latest reasoning log:

```bash
docker exec smart-traffic-system-redis-1 redis-cli XREVRANGE reasoning_logs + - COUNT 1
```

Read the latest phase state:

```bash
docker exec smart-traffic-system-redis-1 redis-cli XREVRANGE phase_states + - COUNT 1
```

## 7. Benchmark a Checkpoint

```bash
uv run python -m benchmark.compare ^
  --checkpoint checkpoints/dqn_ew_repair_best.pt ^
  --sumocfg simulation/net/intersection.sumocfg ^
  --episode-duration 300 ^
  --seeds 1 2 3 ^
  --backend traci
```

Benchmark reports are written to:

```text
benchmark/results/
```

## Troubleshooting

### Docker build cannot find the checkpoint

Make sure this file exists before starting the full stack:

```text
checkpoints/dqn_ew_repair_best.pt
```

The compose file mounts `./checkpoints` into the agent container at runtime.

### Agent exits after the episode

That is expected. The default agent command runs one SUMO episode.

To run another episode:

```bash
docker compose up agent
```

### Docker says Redis is already running

That is fine. Check:

```bash
docker compose ps
```

### Docker says a port is already allocated

Override the host-facing port and start again:

```bash
DASHBOARD_HOST_PORT=8503 docker compose up -d dashboard
API_HOST_PORT=8001 docker compose up -d api
REDIS_HOST_PORT=6380 docker compose up -d redis
```

### SUMO cannot be found

For Docker mode, SUMO is installed inside `docker/app/Dockerfile`.

For local development mode, install SUMO and either add its `bin` directory to `PATH` or set `SUMO_HOME`.

Example Windows path:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

### Dashboard connects but charts are empty

Make sure at least one producer is publishing:

- `rl.control_runner` for `reasoning_logs` and `agent_commands`
- `control.service` for `phase_states`
- `vision.producer.smoke_test` or `vision.producer.main` for `vehicle_counts`

### Control service stays all-red

Confirm that `agent_commands` is increasing:

```bash
docker exec smart-traffic-system-redis-1 redis-cli XLEN agent_commands
```

If it is not increasing, start `rl.control_runner` or `tests.mock_agent`.
