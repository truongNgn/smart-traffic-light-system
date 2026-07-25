"""12-factor config: every runtime knob comes from env vars, never hardcoded."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.constants import DEFAULT_SEED, DEFAULT_STEP_LENGTH_S


class SimulationSettings(BaseSettings):
    """Settings for the SUMO/TraCI simulation side (Engineer B)."""

    model_config = SettingsConfigDict(env_prefix="SIM_", env_file=".env", extra="ignore")

    sumo_home: str | None = Field(
        default=None,
        description="Path to the SUMO install (SUMO_HOME). Falls back to the env var of the same name.",
    )
    sumocfg_path: str = Field(
        default="simulation/net/intersection.sumocfg",
        description="Path to the .sumocfg used to launch SUMO.",
    )
    use_gui: bool = Field(default=False, description="Launch sumo-gui instead of headless sumo.")
    seed: int = Field(default=DEFAULT_SEED)
    step_length_s: float = Field(default=DEFAULT_STEP_LENGTH_S)
    traci_port: int | None = Field(
        default=None, description="Fixed TraCI port; None lets traci pick a free one."
    )
    backend: Literal["traci", "libsumo"] = Field(
        default="traci",
        description="'traci' (subprocess+socket, supports GUI) or 'libsumo' "
        "(in-process, ~8x faster stepping, headless only - see "
        "simulation/traci_wrapper/session.py).",
    )


settings = SimulationSettings()
