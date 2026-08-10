"""Redis Streams implementation of the MessageBus Protocol."""

import asyncio
import json
from typing import Any, Awaitable, Callable

import redis as sync_redis
import redis.asyncio as redis
import structlog
from pydantic import BaseModel

from common.config import redis_settings

logger = structlog.get_logger("redis_bus")


class RedisMessageBus:
    def __init__(self, host: str = redis_settings.host, port: int = redis_settings.port):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self._running = False

    async def publish(self, topic: str, key: str, event: BaseModel) -> None:
        """Publish a Pydantic model event to a Redis Stream.
        Note: Redis streams don't strictly partition by key like Kafka, but we keep the key in the signature for compatibility.
        """
        payload = {"data": event.model_dump_json()}
        try:
            await self.redis.xadd(topic, payload)
        except Exception as e:
            logger.error(f"Failed to publish to {topic}", error=str(e))
            raise

    async def subscribe(
        self,
        topic: str,
        consumer_group: str,
        consumer_name: str,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        """Subscribe to a Redis stream with consumer group semantics."""
        self._running = True
        
        # Ensure consumer group exists
        try:
            await self.redis.xgroup_create(topic, consumer_group, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Failed to create consumer group {consumer_group} for {topic}", error=str(e))
                if on_error:
                    await on_error(e)
                raise
        
        logger.info(f"Subscribed to {topic} as {consumer_group}:{consumer_name}")
        
        while self._running:
            try:
                streams = await self.redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {topic: ">"},
                    count=10,
                    block=1000
                )
                
                if not streams:
                    continue
                
                for stream_name, messages in streams:
                    for message_id, data in messages:
                        if "data" in data:
                            payload = json.loads(data["data"])
                            await callback(payload)
                            # Explicit ack
                            await self.redis.xack(topic, consumer_group, message_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Redis read error", error=str(e))
                if on_error:
                    await on_error(e)
                await asyncio.sleep(1.0)

    async def close(self):
        self._running = False
        await self.redis.aclose()


class RedisStreamBus:
    """Synchronous MessageBus implementation using Redis Streams for backward compatibility with Vision."""

    def __init__(self, host: str = redis_settings.host, port: int = redis_settings.port, db: int = 0) -> None:
        self._redis = sync_redis.Redis(host=host, port=port, db=db, decode_responses=True)
        try:
            self._redis.ping()
            logger.info("Connected to Redis", host=host, port=port)
        except sync_redis.ConnectionError as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise

    def publish(self, stream_name: str, event: Any) -> None:
        """
        Publish a Pydantic model to Redis Stream.
        The event is dumped to JSON and stored under the 'data' key.
        """
        data_json: str = event.model_dump_json()
        payload: dict[str, str] = {"data": data_json}

        try:
            msg_id = self._redis.xadd(stream_name, payload)
            logger.debug("Published event", stream_name=stream_name, msg_id=msg_id)
        except Exception as e:
            logger.error("Failed to publish event", error=str(e), stream_name=stream_name)

    def subscribe(self, stream_name: str, consumer_group: str, consumer_name: str) -> Any:
        raise NotImplementedError("Subscribe logic is implemented in the async RedisMessageBus.")
