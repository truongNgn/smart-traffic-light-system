"""Message Bus Protocol (Engineer A)."""

from __future__ import annotations

from typing import Protocol, Any, Dict


class MessageBus(Protocol):
    """Protocol defining the interface for the event bus."""

    def publish(self, stream_name: str, event: Any) -> None:
        """Publish a Pydantic model event to a stream."""
        ...

    def subscribe(self, stream_name: str, consumer_group: str, consumer_name: str) -> Any:
        """Subscribe to a stream (to be implemented later)."""
        ...
