"""Finite State Machine for traffic light phase transitions."""

import asyncio
import time
from typing import Optional, Callable, Awaitable

import structlog

from common.constants import Direction, YELLOW_DURATION_S, ALL_RED_DURATION_S
from common.schemas.control import PhaseState

logger = structlog.get_logger("phase_fsm")


class PhaseFSM:
    def __init__(self, publish_cb: Callable[[PhaseState], Awaitable[None]]):
        self.current_direction: Optional[Direction] = None
        self.state: str = "ALL_RED"
        self.publish_cb = publish_cb
        self._lock = asyncio.Lock()

    def get_state(self) -> PhaseState:
        return PhaseState(
            active_direction=self.current_direction,
            is_yellow=(self.state == "YELLOW"),
            is_all_red=(self.state == "ALL_RED"),
            timestamp_s=time.time()
        )

    async def _publish(self):
        await self.publish_cb(self.get_state())

    async def force_all_red(self):
        """Emergency or watchdog fallback to all red."""
        async with self._lock:
            if self.state == "ALL_RED":
                return
            
            logger.warning("Forcing ALL_RED state")
            if self.state == "GREEN":
                self.state = "YELLOW"
                await self._publish()
                await asyncio.sleep(YELLOW_DURATION_S)
                
            self.state = "ALL_RED"
            self.current_direction = None
            await self._publish()

    async def transition_to(self, target: Direction):
        """Safely transition to the target direction."""
        async with self._lock:
            if self.current_direction == target and self.state == "GREEN":
                # No-op fast path
                return

            logger.info("FSM Transitioning", current=self.current_direction, target=target)

            if self.state == "GREEN" and self.current_direction is not None:
                self.state = "YELLOW"
                await self._publish()
                await asyncio.sleep(YELLOW_DURATION_S)

            if self.state != "ALL_RED":
                self.state = "ALL_RED"
                self.current_direction = None
                await self._publish()
                await asyncio.sleep(ALL_RED_DURATION_S)

            self.state = "GREEN"
            self.current_direction = target
            await self._publish()
            logger.info("FSM Reached Target", target=target)
