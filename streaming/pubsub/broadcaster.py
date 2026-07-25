"""Async Broadcaster for fanning out events to WebSockets."""

import asyncio
from typing import Set

import structlog
from fastapi import WebSocket

from common.schemas.vision import VehicleCountEvent

logger = structlog.get_logger("broadcaster")


class Broadcaster:
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self._queue: asyncio.Queue[VehicleCountEvent] = asyncio.Queue()
        self._task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)
        logger.info("Client connected", active_clients=len(self.connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
            logger.info("Client disconnected", active_clients=len(self.connections))

    async def broadcast(self, event: VehicleCountEvent):
        """Put event into the internal queue for async broadcasting."""
        await self._queue.put(event)

    async def _broadcast_loop(self):
        """Background task that reads from the queue and sends to all clients."""
        logger.info("Started broadcast loop")
        while True:
            try:
                event = await self._queue.get()
                if not self.connections:
                    continue
                
                payload = event.model_dump_json()
                
                # Send to all clients concurrently
                tasks = []
                for connection in list(self.connections):
                    tasks.append(connection.send_text(payload))
                
                if tasks:
                    # Ignore connection errors for disconnected clients
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            pass
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in broadcast loop", error=str(e))

    def start(self):
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
