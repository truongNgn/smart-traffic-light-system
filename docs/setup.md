# Development setup

## Python environment

Use a project-local virtualenv (`.venv/`, gitignored) rather than installing
into your system/user Python:

```bash
python -m venv .venv
.venv/Scripts/pip install pydantic pydantic-settings structlog traci sumolib gymnasium numpy pytest pytest-asyncio ruff mypy
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`torch` is installed separately from the CPU-only wheel index - the default
PyPI index resolves a much larger CUDA build even on machines without a GPU.

Run everything through the venv's interpreter:

```bash
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python -m simulation.traci_wrapper.smoke_test --steps 200
```

## SUMO

Install [SUMO](https://sumo.dlr.de/docs/Downloads.php) and set `SUMO_HOME`
to the install directory (e.g. `C:\Program Files (x86)\Eclipse\Sumo` on
Windows) so `sumo`, `sumo-gui`, and `netconvert` are discoverable - see
[simulation/net/README.md](../simulation/net/README.md) for building the
network and route files.

Tests that need SUMO skip cleanly (not fail) when it isn't installed or the
network hasn't been built - see `tests/conftest.py`.
