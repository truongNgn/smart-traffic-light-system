"""Message Bus Protocol (Engineer A)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel


class MessageBus(Protocol):
    """Protocol defining the interface for the event bus."""

    async def publish(self, topic: str, key: str, event: BaseModel) -> None:
        """Publish a Pydantic model event to a stream/topic."""
        ...

    async def subscribe(
        self,
        topic: str,
        consumer_group: str,
        consumer_name: str,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        """Subscribe to a topic with consumer group semantics."""
        ...

