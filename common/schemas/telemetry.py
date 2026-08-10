"""Telemetry data contracts for system monitoring."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from common.schemas.base import BaseEvent


class Telemetry(BaseEvent):
    """System-level metrics (e.g., FPS, latency) published for monitoring."""

    service_name: str = Field(description="Name of the service emitting telemetry (e.g., 'vision', 'rl_agent').")
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Key-value pairs of metrics (e.g., {'fps': 30.5, 'latency_ms': 12.0})."
    )
