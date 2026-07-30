"""Connection lifecycle around SUMO: find the binary (traci backend) or the
in-process library (libsumo backend), start it, step it, and guarantee
teardown even on exceptions.

Two backends, same interface:

- **traci** (default): SUMO runs as a subprocess, this process talks to it
  over a TCP socket. Supports sumo-gui and multiple concurrent labeled
  connections. The socket round-trip is pure overhead per call though -
  benchmarked at ~8x slower than libsumo for step-heavy workloads (see
  docs/training_on_kaggle.md), which matters a lot over an RL training run.
- **libsumo**: the SUMO engine linked directly into the Python process, no
  socket, no subprocess. Much faster, but headless only (no sumo-gui) and
  only one simulation can be active per process at a time (no `label`
  multiplexing) - use it for training, use `traci` when you want to *watch*
  the simulation (smoke_test.py --gui) or need concurrent sessions.

Downstream code (rl/env/, simulation/state/) should depend on TraciSession
and its `.traci` property, never call `traci.*`/`libsumo.*` directly, so
this stays the only place that knows which backend is active.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import TracebackType
from typing import Literal

from common.config import SimulationSettings
from common.constants import DEFAULT_SEED, DEFAULT_STEP_LENGTH_S
from common.logging import get_logger

logger = get_logger(component="traci_wrapper")

Backend = Literal["traci", "libsumo"]


class SumoNotFoundError(RuntimeError):
    """Raised when neither `sumo`/`sumo-gui`/`libsumo` nor SUMO_HOME can be found."""


def _ensure_traci_importable(sumo_home: str | None) -> None:
    """Make `import traci` work whether it was pip-installed or only ships
    inside a SUMO install's tools/ directory."""
    try:
        import traci  # noqa: F401

        return
    except ImportError:
        pass

    home = sumo_home or os.environ.get("SUMO_HOME")
    if not home:
        raise SumoNotFoundError(
            "traci is not importable and SUMO_HOME is not set. Either "
            "`pip install traci sumolib`, or install SUMO and set SUMO_HOME."
        )
    tools_dir = Path(home) / "tools"
    if not tools_dir.exists():
        raise SumoNotFoundError(f"SUMO_HOME is set to {home!r} but {tools_dir} does not exist.")
    sys.path.insert(0, str(tools_dir))


def _import_libsumo():  # noqa: ANN201 - returns the libsumo module
    try:
        import libsumo

        return libsumo
    except ImportError as exc:
        raise SumoNotFoundError(
            "backend='libsumo' requires the `libsumo` package: `pip install libsumo`."
        ) from exc


def find_sumo_binary(use_gui: bool, sumo_home: str | None = None) -> str:
    name = "sumo-gui" if use_gui else "sumo"
    exe = shutil.which(name)
    if exe:
        return exe

    home = sumo_home or os.environ.get("SUMO_HOME")
    if home:
        candidate = Path(home) / "bin" / name
        for suffix in ("", ".exe"):
            path = candidate.with_suffix(suffix)
            if path.exists():
                return str(path)

    raise SumoNotFoundError(
        f"Could not find the '{name}' executable on PATH or under $SUMO_HOME/bin. "
        "Install SUMO from https://sumo.dlr.de/ and either add its bin/ directory "
        "to PATH or set the SUMO_HOME environment variable."
    )


class TraciSession:
    """Context manager around one SUMO simulation run.

        with TraciSession(sumocfg_path="simulation/net/intersection.sumocfg") as sim:
            for _ in range(3600):
                sim.step()
                vehicle_ids = sim.traci.vehicle.getIDList()

    Multiple concurrent sessions (e.g. parallel RL rollout workers) must pass
    distinct `label` values when backend="traci" - TraCI multiplexes
    connections by label. backend="libsumo" has no such concept - only one
    session may be active per process at a time.
    """

    def __init__(
        self,
        sumocfg_path: str | Path,
        *,
        use_gui: bool = False,
        seed: int = DEFAULT_SEED,
        step_length_s: float = DEFAULT_STEP_LENGTH_S,
        sumo_home: str | None = None,
        port: int | None = None,
        label: str = "default",
        extra_args: list[str] | None = None,
        backend: Backend = "traci",
    ) -> None:
        if backend == "libsumo" and use_gui:
            raise ValueError("backend='libsumo' does not support use_gui=True; use backend='traci'.")

        self.sumocfg_path = Path(sumocfg_path)
        self.use_gui = use_gui
        self.seed = seed
        self.step_length_s = step_length_s
        self.sumo_home = sumo_home
        self.port = port
        self.label = label
        self.extra_args = extra_args or []
        self.backend: Backend = backend

        self._traci_module = None
        self._connected = False
        self._step_count = 0

    @classmethod
    def from_settings(cls, settings: SimulationSettings, **overrides: object) -> "TraciSession":
        kwargs = dict(
            sumocfg_path=settings.sumocfg_path,
            use_gui=settings.use_gui,
            seed=settings.seed,
            step_length_s=settings.step_length_s,
            sumo_home=settings.sumo_home,
            port=settings.traci_port,
            backend=settings.backend,
        )
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]

    def _build_args(self) -> list[str]:
        """The command-line args after the binary itself - shared by both
        backends (libsumo ignores the binary path element entirely, so it
        gets a placeholder instead of a real resolved executable)."""
        if not self.sumocfg_path.exists():
            raise FileNotFoundError(f"sumocfg not found: {self.sumocfg_path}")
        args = [
            "-c", str(self.sumocfg_path),
            "--seed", str(self.seed),
            "--step-length", str(self.step_length_s),
            "--start" if self.use_gui else "--no-step-log",
            "true",
        ]
        if self.use_gui:
            # Without this, sumo-gui pops up a blocking "Simulation ended -
            # close all open files and views?" dialog once TraCI ends the
            # episode, and the whole process (not just the GUI) hangs until
            # a human clicks it - fatal for any unattended/automated run.
            args.append("--quit-on-end")
        args.extend(self.extra_args)
        return args

    def _build_command(self) -> list[str]:
        binary = find_sumo_binary(self.use_gui, self.sumo_home)
        return [binary] + self._build_args()

    def start(self) -> "TraciSession":
        if self._connected:
            raise RuntimeError("TraciSession is already connected.")

        if self.backend == "libsumo":
            libsumo = _import_libsumo()
            args = self._build_args()
            cmd = ["sumo"] + args  # placeholder binary path - libsumo ignores it
            logger.info("sumo.start", cmd=cmd, label=self.label, seed=self.seed, backend="libsumo")
            libsumo.start(cmd, label=self.label)
            self._traci_module = libsumo
        else:
            _ensure_traci_importable(self.sumo_home)
            import traci

            cmd = self._build_command()
            logger.info("sumo.start", cmd=cmd, label=self.label, seed=self.seed, backend="traci")
            traci.start(cmd, port=self.port, label=self.label)
            self._traci_module = traci.getConnection(self.label)

        self._connected = True
        return self

    def step(self, n: int = 1) -> int:
        """Advance the simulation by n steps. Returns the new step count."""
        if not self._connected:
            raise RuntimeError("Session not started; use `with TraciSession(...) as sim:`.")
        if self._traci_module is None:
            raise RuntimeError("Session has no active TraCI connection.")
        for _ in range(n):
            self._traci_module.simulationStep()
            self._step_count += 1
        return self._step_count

    def close(self) -> None:
        if self._connected and self._traci_module is not None:
            try:
                self._traci_module.close()
            except Exception:  # noqa: BLE001 - best-effort teardown, never mask the original error
                logger.warning("sumo.close_failed", label=self.label)
            finally:
                self._connected = False
                logger.info("sumo.closed", label=self.label, steps=self._step_count)

    @property
    def traci(self):  # noqa: ANN201 - returns the live traci connection or libsumo module
        if not self._connected:
            raise RuntimeError("Session not started.")
        return self._traci_module

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self) -> "TraciSession":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
