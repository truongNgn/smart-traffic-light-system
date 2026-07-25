"""Telemetry data contracts for system monitoring."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class Telemetry(BaseModel):
    """System-level metrics (e.g., FPS, latency) published for monitoring."""

    service_name: str = Field(description="Name of the service emitting telemetry (e.g., 'vision', 'rl_agent').")
    timestamp_s: float = Field(description="Timestamp in seconds.")
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Key-value pairs of metrics (e.g., {'fps': 30.5, 'latency_ms': 12.0})."
    )
