"""Control daemon that listens to AI commands and drives the FSM."""

import asyncio
import json
import time

import redis.asyncio as redis
import structlog
from pydantic import ValidationError

from common.config import redis_settings
from common.schemas.control import PhaseAction, PhaseState
from control.fsm import PhaseFSM

logger = structlog.get_logger("control_service")

class ControlService:
    def __init__(self):
        self.redis = redis.Redis(
            host=redis_settings.host, 
            port=redis_settings.port, 
            decode_responses=True
        )
        self.fsm = PhaseFSM(publish_cb=self.publish_state)
        self._running = False
        self.last_command_time = time.time()
        self.watchdog_timeout_s = 10.0

    async def publish_state(self, state: PhaseState):
        """Callback for the FSM to publish state updates to Redis."""
        payload = {"data": state.model_dump_json()}
        try:
            await self.redis.xadd("phase_states", payload)
            logger.debug("Published PhaseState", state=state.model_dump())
        except Exception as e:
            logger.error("Failed to publish PhaseState", error=str(e))

    async def watchdog(self):
        """Monitors for AI stalling and fails safe to ALL_RED."""
        while self._running:
            await asyncio.sleep(1.0)
            if time.time() - self.last_command_time > self.watchdog_timeout_s:
                if self.fsm.state != "ALL_RED":
                    logger.error("Watchdog timeout! No commands received. Failing safe.")
                    await self.fsm.force_all_red()

    async def consume_commands(self):
        """Listens for AI PhaseActions on Redis."""
        last_id = "$"
        logger.info("Listening for agent commands on 'agent_commands'")
        
        while self._running:
            try:
                streams = await self.redis.xread({"agent_commands": last_id}, count=1, block=1000)
                if not streams:
                    continue
                    
                # Untyped unpack for brevity, we know the structure
                for stream_name, messages in streams:
                    for message_id, data in messages:
                        last_id = message_id
                        if "data" in data:
                            try:
                                payload = json.loads(data["data"])
                                action = PhaseAction(**payload)
                                self.last_command_time = time.time()
                                logger.info("Received PhaseAction", action=action.model_dump())
                                
                                # Do not await the transition directly in the read loop if you want 
                                # to keep reading, but FSM has a lock so we can just fire it as a task.
                                asyncio.create_task(self.fsm.transition_to(action.target_direction))
                                
                            except ValidationError as e:
                                logger.error("Invalid PhaseAction", error=str(e))
                            except Exception as e:
                                logger.error("Failed to process command", error=str(e))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Redis read error", error=str(e))
                await asyncio.sleep(1.0)

    async def run(self):
        self._running = True
        logger.info("Control Service starting")
        
        # Publish initial state
        await self.publish_state(self.fsm.get_state())
        
        watchdog_task = asyncio.create_task(self.watchdog())
        consume_task = asyncio.create_task(self.consume_commands())
        
        try:
            await asyncio.gather(watchdog_task, consume_task)
        except asyncio.CancelledError:
            self._running = False
            watchdog_task.cancel()
            consume_task.cancel()
            await self.redis.aclose()
            logger.info("Control Service stopped")

if __name__ == "__main__":
    service = ControlService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        pass
