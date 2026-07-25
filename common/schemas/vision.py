"""Vision-related data contracts (Engineer A)."""

from __future__ import annotations

from typing import Dict, Tuple

from pydantic import BaseModel, Field


class FeedFrame(BaseModel):
    """A raw frame (metadata) captured from a camera feed."""

    camera_id: str = Field(description="Unique identifier for the camera/feed (e.g., 'N', 'S', 'E', 'W').")
    timestamp_s: float = Field(description="Capture timestamp in seconds since epoch.")
    frame_shape: Tuple[int, int, int] = Field(description="Shape of the frame array (height, width, channels).")


class VehicleCountEvent(BaseModel):
    """The enriched event emitted by the vision pipeline for one frame."""

    camera_id: str = Field(description="Originating camera ID.")
    timestamp_s: float = Field(description="Capture timestamp in seconds.")
    class_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Dictionary mapping vehicle class (e.g., 'car', 'bus') to count.",
    )
    centroids: Dict[int, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Dictionary mapping track ID to (x, y) centroid coordinates.",
    )
