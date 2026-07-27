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


class VisionSettings(BaseSettings):
    """Settings for the YOLO Vision pipeline (Engineer A)."""

    model_config = SettingsConfigDict(env_prefix="VISION_", env_file=".env", extra="ignore")

    yolo_model_path: str = Field(
        default="yolov8n.pt",
        description="Path or name of the YOLOv8 model.",
    )
    camera_fps: float = Field(
        default=30.0,
        description="Target FPS for the video stream simulator.",
    )
    conf_threshold: float = Field(
        default=0.25,
        description="Confidence threshold for YOLO detection.",
    )
    iou_threshold: float = Field(
        default=0.45,
        description="NMS IoU threshold for YOLO detection.",
    )
    # A generic mapping of lane/ROI name to polygon points [(x,y), ...]
    # For now, we will use hardcoded defaults that cover standard frames,
    # or rely on speed calculation across the whole frame if ROIs aren't strictly defined.
    rois: dict[str, list[tuple[int, int]]] = Field(
        default={
            "Lane-Center": [(250, 0), (390, 0), (390, 480), (250, 480)],
        },
        description="Dictionary mapping lane ID to a list of (x,y) polygon points.",
    )



class RedisSettings(BaseSettings):
    """Settings for the Redis Message Bus."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    stream_name: str = Field(default="vehicle_counts")


class ApiSettings(BaseSettings):
    """Settings for the FastAPI application (Engineer A)."""

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: list[str] = Field(default=["*"])


settings = SimulationSettings()
vision_settings = VisionSettings()
redis_settings = RedisSettings()
api_settings = ApiSettings()
