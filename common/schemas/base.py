"""Base schema for all events sent over the message bus."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from common.config import site_settings


def _get_intersection_id() -> str:
    return site_settings.intersection_id

def _get_site_id() -> str:
    return site_settings.site_id

def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseEvent(BaseModel):
    """Base schema for all events, ensuring spatial identity and timestamping."""

    intersection_id: str = Field(default_factory=_get_intersection_id, description="Unique ID for this intersection (e.g., 'hcm.q1.nvc-lelai').")
    site_id: str = Field(default_factory=_get_site_id, description="Cluster or region ID used for routing.")
    schema_version: int = Field(default=1, description="Version of the event schema.")
    event_ts: datetime = Field(default_factory=_now, description="Timestamp when the event was produced at the source.")
    ingest_ts: datetime | None = Field(
        default=None, 
        description="Timestamp when the bus received it. Measures end-to-end latency."
    )
