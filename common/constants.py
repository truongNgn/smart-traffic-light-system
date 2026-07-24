"""Single source of truth for every number the paper's methodology pins down.

Both the RL environment (rl/env/) and the production control executor
(control/) must import phase-timing constants from here. Defining the same
number twice in two packages is how a trained policy ends up unsafe in
deployment — see docs/adr/0002-reward-and-safety-buffers.md.
"""

from __future__ import annotations

from enum import IntEnum

# --- Intersection geometry (Sahal et al. 2023) ---------------------------
APPROACH_LENGTH_M: float = 100.0
"""Length of each of the 4 approach segments feeding the intersection."""

GRID_CELLS_TOTAL: int = 80
"""Total discretized cells across all 4 approaches (the DQN input size)."""

GRID_CELLS_PER_APPROACH: int = GRID_CELLS_TOTAL // 4
"""20 cells per approach, each APPROACH_LENGTH_M / GRID_CELLS_PER_APPROACH long."""

CELL_LENGTH_M: float = APPROACH_LENGTH_M / GRID_CELLS_PER_APPROACH

# --- Vehicle classes ------------------------------------------------------
class VehicleClass(IntEnum):
    CAR = 0
    MOTORCYCLE = 1
    BUS = 2
    TRUCK = 3


VEHICLE_CLASS_TO_SUMO_VCLASS: dict[VehicleClass, str] = {
    VehicleClass.CAR: "passenger",
    VehicleClass.MOTORCYCLE: "motorcycle",
    VehicleClass.BUS: "bus",
    VehicleClass.TRUCK: "truck",
}

# --- Action space ----------------------------------------------------------
class Direction(IntEnum):
    """4 discrete actions: which approach gets the green phase."""

    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3


NUM_ACTIONS: int = len(Direction)

# --- Mandatory phase-switch safety buffers --------------------------------
YELLOW_DURATION_S: float = 2.0
ALL_RED_DURATION_S: float = 2.0

# --- Waiting-time / reward -------------------------------------------------
STOPPED_SPEED_THRESHOLD_MPS: float = 0.1
"""A vehicle counts toward accumulated waiting time below this speed."""

# --- DQN architecture -------------------------------------------------------
DQN_INPUT_SIZE: int = GRID_CELLS_TOTAL
DQN_HIDDEN_LAYERS: int = 5
DQN_HIDDEN_SIZE: int = 400
DQN_OUTPUT_SIZE: int = NUM_ACTIONS

# --- Simulation defaults ----------------------------------------------------
DEFAULT_STEP_LENGTH_S: float = 1.0
DEFAULT_SEED: int = 42
