"""Control daemon that listens to AI commands and drives the FSM."""

import asyncio
import time

import structlog
from pydantic import ValidationError

from common.config import site_settings
from common.schemas.control import PhaseAction, PhaseState
from control.fsm import PhaseFSM
from streaming.bus.factory import get_message_bus

logger = structlog.get_logger("control_service")

class ControlService:
    def __init__(self):
        self.bus = get_message_bus()
        self.fsm = PhaseFSM(publish_cb=self.publish_state)
        self._running = False
        self.last_command_time = time.time()
        self.watchdog_timeout_s = 10.0

    async def publish_state(self, state: PhaseState):
        """Callback for the FSM to publish state updates to the bus."""
        try:
            await self.bus.publish("traffic.phase_state.v1", site_settings.intersection_id, state)
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

    async def on_command(self, payload: dict) -> None:
        try:
            action = PhaseAction(**payload)
            self.last_command_time = time.time()
            logger.info("Received PhaseAction", action=action.model_dump())
            asyncio.create_task(self.fsm.transition_to(action.target_phase))
        except ValidationError as e:
            logger.error("Invalid PhaseAction", error=str(e))
        except Exception as e:
            logger.error("Failed to process command", error=str(e))

    async def consume_commands(self):
        """Listens for AI PhaseActions on the bus."""
        logger.info("Listening for agent commands on 'traffic.commands.v1'")
        while self._running:
            try:
                await self.bus.subscribe(
                    topic="traffic.commands.v1",
                    consumer_group="control_group",
                    consumer_name="control_worker_1",
                    callback=self.on_command
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bus read error", error=str(e))
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
            await getattr(self.bus, "close", lambda: asyncio.sleep(0))()
            logger.info("Control Service stopped")

if __name__ == "__main__":
    service = ControlService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        pass
