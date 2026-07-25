"""Async consumer for Redis Streams to bridge bus to API."""

import asyncio
import json
from typing import AsyncGenerator

import redis.asyncio as redis
import structlog

from common.schemas.vision import VehicleCountEvent

logger = structlog.get_logger("redis_consumer")


class RedisConsumer:
    def __init__(self, host: str, port: int, stream_name: str):
        self.host = host
        self.port = port
        self.stream_name = stream_name
        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        self._running = False

    async def connect(self) -> None:
        await self.client.ping()
        logger.info("Connected to Redis (Consumer)", host=self.host, port=self.port)

    async def close(self) -> None:
        self._running = False
        await self.client.aclose()
        logger.info("Disconnected from Redis (Consumer)")

    async def listen(self, last_id: str = "$") -> AsyncGenerator[VehicleCountEvent, None]:
        """Listen to the stream and yield events as they arrive."""
        self._running = True
        logger.info("Listening to stream", stream=self.stream_name, starting_id=last_id)
        
        while self._running:
            try:
                # Block for up to 1 second
                streams = await self.client.xread({self.stream_name: last_id}, count=10, block=1000)
                if not streams:
                    continue
                    
                from typing import cast, List, Tuple, Dict
                streams_typed = cast(List[Tuple[str, List[Tuple[str, Dict[str, str]]]]], streams)
                    
                for stream_name, messages in streams_typed:
                    for message_id, data in messages:
                        last_id = message_id
                        if "event" in data:
                            try:
                                payload = json.loads(data["event"])
                                event = VehicleCountEvent(**payload)
                                yield event
                            except Exception as e:
                                logger.error("Failed to parse event", error=str(e), data=data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error reading from Redis stream", error=str(e))
                await asyncio.sleep(1)
