"""Connection lifecycle around TraCI: find the SUMO binary, start it as a
subprocess, step it, and guarantee teardown even on exceptions.

TraCI itself is a client-server protocol - SUMO runs as the server, this
process is the client - so "wrapping TraCI" mostly means: locating the right
binary, building the command line correctly, and never leaking a live SUMO
process when something goes wrong. Downstream code (rl/env/, simulation/state/)
should depend on TraciSession, not call `traci.*` directly, so the connection
details stay in one place.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import TracebackType

from common.config import SimulationSettings
from common.constants import DEFAULT_SEED, DEFAULT_STEP_LENGTH_S
from common.logging import get_logger

logger = get_logger(component="traci_wrapper")


class SumoNotFoundError(RuntimeError):
    """Raised when neither `sumo`/`sumo-gui` nor SUMO_HOME can locate a binary."""


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
    distinct `label` values - TraCI multiplexes connections by label.
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
    ) -> None:
        self.sumocfg_path = Path(sumocfg_path)
        self.use_gui = use_gui
        self.seed = seed
        self.step_length_s = step_length_s
        self.sumo_home = sumo_home
        self.port = port
        self.label = label
        self.extra_args = extra_args or []

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
        )
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]

    def _build_command(self) -> list[str]:
        if not self.sumocfg_path.exists():
            raise FileNotFoundError(f"sumocfg not found: {self.sumocfg_path}")
        binary = find_sumo_binary(self.use_gui, self.sumo_home)
        cmd = [
            binary,
            "-c", str(self.sumocfg_path),
            "--seed", str(self.seed),
            "--step-length", str(self.step_length_s),
            "--start" if self.use_gui else "--no-step-log",
            "true",
        ]
        cmd.extend(self.extra_args)
        return cmd

    def start(self) -> "TraciSession":
        if self._connected:
            raise RuntimeError("TraciSession is already connected.")
        _ensure_traci_importable(self.sumo_home)
        import traci

        cmd = self._build_command()
        logger.info("sumo.start", cmd=cmd, label=self.label, seed=self.seed)
        traci.start(cmd, port=self.port, label=self.label)
        self._traci_module = traci.getConnection(self.label)
        self._connected = True
        return self

    def step(self, n: int = 1) -> int:
        """Advance the simulation by n steps. Returns the new step count."""
        if not self._connected:
            raise RuntimeError("Session not started; use `with TraciSession(...) as sim:`.")
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
    def traci(self):  # noqa: ANN201 - returns the live traci connection module
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
