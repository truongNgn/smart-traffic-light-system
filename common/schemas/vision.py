"""Vision-related data contracts (Engineer A)."""

from __future__ import annotations

from typing import Dict, Tuple

from pydantic import BaseModel, Field

from common.schemas.base import BaseEvent


class FeedFrame(BaseEvent):
    """A raw frame (metadata) captured from a camera feed."""

    camera_id: str = Field(description="Unique identifier for the camera/feed (e.g., 'N', 'S', 'E', 'W').")
    frame_shape: Tuple[int, int, int] = Field(description="Shape of the frame array (height, width, channels).")


class VehicleCountEvent(BaseEvent):
    """The enriched event emitted by the vision pipeline for one frame."""

    camera_id: str = Field(description="Originating camera ID.")
    class_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Dictionary mapping vehicle class (e.g., 'car', 'bus') to count.",
    )
    lane_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Dictionary mapping lane ID (e.g., 'E-Left', 'E-Straight') to current count.",
    )
    mean_speed_mps: float = Field(
        default=0.0,
        description="Estimated mean speed of all tracked vehicles in meters per second.",
    )
    centroids: Dict[int, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Dictionary mapping track ID to (x, y) centroid coordinates.",
    )
