"""Finite State Machine for traffic light phase transitions."""

import asyncio
import time
from typing import Optional, Callable, Awaitable

import structlog

from common.constants import PHASE_DIRECTIONS, YELLOW_DURATION_S, ALL_RED_DURATION_S
from common.constants import PhaseAction as PhaseActionEnum
from common.schemas.control import PhaseState

logger = structlog.get_logger("phase_fsm")


class PhaseFSM:
    def __init__(self, publish_cb: Callable[[PhaseState], Awaitable[None]]):
        self.current_phase: Optional[PhaseActionEnum] = None
        self.state: str = "ALL_RED"
        self.publish_cb = publish_cb
        self._lock = asyncio.Lock()

    def get_state(self) -> PhaseState:
        active_directions = (
            list(PHASE_DIRECTIONS[self.current_phase])
            if self.current_phase is not None and self.state != "ALL_RED"
            else []
        )
        return PhaseState(
            active_phase=self.current_phase if self.state != "ALL_RED" else None,
            active_directions=active_directions,
            active_direction=active_directions[0] if active_directions else None,
            is_yellow=(self.state == "YELLOW"),
            is_all_red=(self.state == "ALL_RED")
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
            self.current_phase = None
            await self._publish()

    async def transition_to(self, target: PhaseActionEnum):
        """Safely transition to the target phase."""
        async with self._lock:
            if self.current_phase == target and self.state == "GREEN":
                # No-op fast path
                return

            logger.info("FSM Transitioning", current=self.current_phase, target=target)

            if self.state == "GREEN" and self.current_phase is not None:
                self.state = "YELLOW"
                await self._publish()
                await asyncio.sleep(YELLOW_DURATION_S)

            if self.state != "ALL_RED":
                self.state = "ALL_RED"
                self.current_phase = None
                await self._publish()
                await asyncio.sleep(ALL_RED_DURATION_S)

            self.state = "GREEN"
            self.current_phase = target
            await self._publish()
            logger.info("FSM Reached Target", target=target)
