# Contributing

This project is a production-style monorepo for a SUMO + DQN smart traffic
control system. Keep changes small, tested, and aligned with the shared
contracts in `common/`.

## Development Setup

```bash
uv sync --dev
```

SUMO is required for the integration tests. On Linux CI it is installed with
`apt-get install sumo sumo-tools`; on Windows, install SUMO and set `SUMO_HOME`
so `sumolib` can find the tools.

## Before Opening a PR

Run:

```bash
uv run ruff check .
uv run mypy common rl simulation
uv run pytest -q --basetemp=.pytest-tmp
```

If you touch Dockerfiles or `docker-compose.yml`, also run a Docker build or a
short compose smoke test.

## Contracts

`common/constants.py` and `common/schemas/` are the system treaty between the
vision/backend side and the SUMO/RL/control side. Any change there should update
tests, docs, and the PR's contract-change checklist.

The canonical action space is two phases:

- `EAST_WEST`
- `NORTH_SOUTH`

Legacy `target_direction` payloads are accepted only as a compatibility path and
map to the owning two-way phase.

## Commit Style

Prefer Conventional Commit prefixes such as `feat:`, `fix:`, `test:`, `docs:`,
and `chore:`. Keep branches short-lived and merge only with a green CI run.
