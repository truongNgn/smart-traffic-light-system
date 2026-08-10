"""12-factor config: every runtime knob comes from env vars, never hardcoded."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.constants import DEFAULT_SEED, DEFAULT_STEP_LENGTH_S
from common.constants import GREEN_DURATION_S


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
        default="yolov8_vehicle_detection.pt",
        description="Path or name of the YOLOv8 model.",
    )
    stream_url: str | None = Field(
        default=None,
        description="Optional RTSP stream URL. If provided, vision will read from this stream instead of a local file.",
    )
    streams: str | None = Field(
        default=None,
        description=(
            "Optional comma-separated camera stream map, for example "
            "'N=rtsp://mediamtx:8554/cam1,S=rtsp://mediamtx:8554/cam2'. "
            "When set, one vision container can process multiple cameras."
        ),
    )
    camera_id: str = Field(
        default="N",
        description="ID of the camera (e.g. N, S, E, W). Used by the agent to determine the origin of the vehicle counts.",
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
    rois: dict[str, list[tuple[float, float]]] = Field(
        default={
            # Camera đặt ở cột đèn, nhìn ngược dòng xe đi tới.
            # Giao thông đi bên phải (Việt Nam) -> dòng xe tiến lại gần camera sẽ nằm ở NỬA TRÁI màn hình.
            "Queue-Zone": [(0.05, 0.95), (0.25, 0.25), (0.50, 0.25), (0.50, 0.95)]
        },
        description="Dictionary mapping lane ID to a list of relative (x,y) polygon points [0.0 - 1.0].",
    )



class RedisSettings(BaseSettings):
    """Settings for the Redis Message Bus."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    stream_name: str = Field(default="traffic.counts.v1")


class ApiSettings(BaseSettings):
    """Settings for the FastAPI application (Engineer A)."""

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: list[str] = Field(default=["*"])


class AgentRuntimeSettings(BaseSettings):
    """Settings for the trained DQN runtime that publishes live decisions."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    checkpoint_path: str = Field(default="checkpoints/dqn_ew_repair_best.pt")
    episode_duration_s: float = Field(default=300.0)
    seed: int = Field(default=DEFAULT_SEED)
    backend: Literal["traci", "libsumo"] = Field(default="traci")
    use_gui: bool = Field(default=False)
    decision_interval_s: float = Field(
        default=GREEN_DURATION_S,
        ge=0.0,
        description="Wall-clock delay between live control decisions. Use 0 for fast smoke tests.",
    )


class SiteSettings(BaseSettings):
    """Global identifiers for the intersection site."""
    model_config = SettingsConfigDict(env_prefix="SITE_", env_file=".env", extra="ignore")

    intersection_id: str = Field(default="default-intersection")
    site_id: str = Field(default="default-site")


settings = SimulationSettings()
vision_settings = VisionSettings()
redis_settings = RedisSettings()
api_settings = ApiSettings()
agent_runtime_settings = AgentRuntimeSettings()
site_settings = SiteSettings()
