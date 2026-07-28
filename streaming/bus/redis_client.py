"""Redis Streams Implementation (Engineer A)."""

from __future__ import annotations

import json
from typing import Any

import redis

from common.logging import get_logger
from streaming.bus.interface import MessageBus

logger = get_logger(component="redis_bus")


class RedisStreamBus(MessageBus):
    """MessageBus implementation using Redis Streams."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        try:
            self._redis.ping()
            logger.info("Connected to Redis", host=host, port=port)
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise

    def publish(self, stream_name: str, event: Any) -> None:
        """
        Publish a Pydantic model to Redis Stream.
        The event is dumped to JSON and stored under the 'data' key.
        """
        # We assume event is a Pydantic model
        data_json: str = event.model_dump_json()
        payload: dict[str, str] = {"data": data_json}

        try:
            msg_id = self._redis.xadd(stream_name, payload)
            logger.debug("Published event", stream_name=stream_name, msg_id=msg_id)
        except Exception as e:
            logger.error("Failed to publish event", error=str(e), stream_name=stream_name)

    def subscribe(self, stream_name: str, consumer_group: str, consumer_name: str) -> Any:
        """Stub for subscription logic (Stage 2)."""
        raise NotImplementedError("Subscribe logic will be implemented in Stage 2.")
