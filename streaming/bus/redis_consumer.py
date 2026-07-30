"""Async consumer for Redis Streams to bridge bus to API."""

import asyncio
from typing import AsyncGenerator, List, Tuple

import redis.asyncio as redis
import structlog

logger = structlog.get_logger("redis_consumer")


class RedisConsumer:
    def __init__(self, host: str, port: int, stream_names: List[str]):
        self.host = host
        self.port = port
        self.stream_names = stream_names
        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        self._running = False

    async def connect(self) -> None:
        await self.client.ping()
        logger.info("Connected to Redis (Consumer)", host=self.host, port=self.port)

    async def close(self) -> None:
        self._running = False
        await self.client.aclose()
        logger.info("Disconnected from Redis (Consumer)")

    async def listen(self, last_ids: dict[str, str] = None) -> AsyncGenerator[Tuple[str, str], None]:
        """Listen to multiple streams and yield (stream_name, raw_json) as they arrive."""
        self._running = True
        
        if last_ids is None:
            last_ids = {name: "$" for name in self.stream_names}
            
        logger.info("Listening to streams", streams=self.stream_names, starting_ids=last_ids)
        
        while self._running:
            try:
                # Block for up to 1 second
                streams = await self.client.xread(last_ids, count=10, block=1000)
                if not streams:
                    continue
                    
                from typing import cast, Dict
                streams_typed = cast(List[Tuple[str, List[Tuple[str, Dict[str, str]]]]], streams)
                    
                for stream_name, messages in streams_typed:
                    for message_id, data in messages:
                        last_ids[stream_name] = message_id
                        if "data" in data:
                            yield stream_name, data["data"]
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error reading from Redis stream", error=str(e))
                await asyncio.sleep(1)
