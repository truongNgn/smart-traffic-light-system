"""Data contracts for intersection control and AI reasoning."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from common.constants import DIRECTION_PHASE, PHASE_DIRECTIONS, Direction
from common.constants import PhaseAction as PhaseActionEnum


class PhaseAction(BaseModel):
    """Command sent by the AI agent to the executor.

    `target_phase` is the canonical model/control contract. `target_direction`
    is accepted for older mock clients and mapped to that direction's two-way
    phase.
    """

    target_phase: PhaseActionEnum = Field(description="The two-way green phase to grant.")
    timestamp_s: float = Field(description="Timestamp of the decision.")

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_target_direction(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "target_phase" in data:
            return {**data, "target_phase": _parse_phase(data["target_phase"])}
        if "target_direction" not in data:
            return data

        direction = _parse_direction(data["target_direction"])
        return {**data, "target_phase": DIRECTION_PHASE[direction]}

    @property
    def target_directions(self) -> tuple[Direction, Direction]:
        return PHASE_DIRECTIONS[self.target_phase]


class PhaseState(BaseModel):
    """The current physical state of the intersection."""

    active_phase: PhaseActionEnum | None = Field(
        description="Which two-way phase currently has right of way. None if all-red."
    )
    active_directions: list[Direction] = Field(
        default_factory=list,
        description="Physical approaches currently receiving green/yellow.",
    )
    active_direction: Direction | None = Field(
        default=None,
        description="Legacy single-direction field; first active direction when present.",
    )
    is_yellow: bool = Field(
        default=False, description="True if the active phase is currently showing yellow."
    )
    is_all_red: bool = Field(
        default=False, description="True if all lights are currently red (safety buffer)."
    )
    timestamp_s: float = Field(description="Timestamp when this state became active.")


class ReasoningLog(BaseModel):
    """Detailed log of the AI's decision process for the dashboard."""

    timestamp_s: float = Field(description="Timestamp of the decision.")
    q_values: dict[str, float] = Field(
        description="Q-values for each possible phase, e.g. {'EAST_WEST': 12.5}."
    )
    chosen_action: PhaseActionEnum = Field(description="The phase selected by the policy.")
    exploration: bool = Field(
        default=False, description="True if the action was chosen randomly (epsilon-greedy)."
    )
    sim_time_s: float | None = Field(default=None, description="Current SUMO simulation time.")
    total_waiting_time_s: float | None = Field(
        default=None, description="Total network waiting time observed by the agent."
    )


def _parse_phase(value: Any) -> PhaseActionEnum:
    if isinstance(value, PhaseActionEnum):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return PhaseActionEnum(int(value))
        return PhaseActionEnum[value]
    return PhaseActionEnum(value)


def _parse_direction(value: Any) -> Direction:
    if isinstance(value, Direction):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return Direction(int(value))
        return Direction[value]
    return Direction(value)
