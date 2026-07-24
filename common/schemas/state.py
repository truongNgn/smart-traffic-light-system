"""The state-side contract between Engineer B's SUMO/TraCI code and the
Stage-3 RL agent. See docs/state_encoding.md for exactly how `grid` is built.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from common.constants import GRID_CELLS_TOTAL


class IntersectionState(BaseModel):
    """One control-step snapshot of the intersection, ready to feed the DQN."""

    step: int = Field(ge=0, description="Simulation step this state was captured at.")
    grid: list[float] = Field(
        description=(
            f"Length-{GRID_CELLS_TOTAL} occupancy vector. Index layout: "
            "4 approaches (East, North, West, South - common.constants.Direction "
            "order) x 20 cells each, cell 0 nearest the stop line. "
            "1.0 = at least one vehicle occupies that 5m cell, 0.0 = empty."
        )
    )
    total_waiting_time_s: float = Field(
        ge=0.0, description="Sum of getWaitingTime() across every vehicle in the network right now."
    )

    @field_validator("grid")
    @classmethod
    def _validate_grid_shape(cls, value: list[float]) -> list[float]:
        if len(value) != GRID_CELLS_TOTAL:
            raise ValueError(f"grid must have exactly {GRID_CELLS_TOTAL} cells, got {len(value)}")
        if any(v not in (0.0, 1.0) for v in value):
            raise ValueError("grid cells must be binary occupancy values (0.0 or 1.0)")
        return value
