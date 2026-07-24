"""Converts raw TraCI vehicle state into the paper's 80-cell grid
representation (Sahal et al. 2023, section on state space). See
docs/state_encoding.md for the full derivation and worked examples - this
docstring covers only the mechanics.

Each of the 4 approach edges (one per Direction) is nominally
APPROACH_LENGTH_M long (netconvert trims the exact length to fit the
junction's real geometry, so GridEncoder queries the true length from TraCI
rather than assuming the constant) and split into GRID_CELLS_PER_APPROACH
equal cells. Cell 0 of an approach is nearest the stop line at the
junction; cell 19 is nearest the network boundary. A cell is 1.0 if any
vehicle (in either lane) currently occupies it, else 0.0. The two lanes of
an approach are collapsed into one set of 20 cells - the paper's grid has
no separate lane dimension.
"""

from __future__ import annotations

from common.constants import (
    CELL_LENGTH_M,
    GRID_CELLS_PER_APPROACH,
    GRID_CELLS_TOTAL,
    Direction,
)
from common.schemas.state import IntersectionState
from simulation.state.waiting_time import total_waiting_time

# Direction -> the incoming edge id vehicles queue on before that direction's
# green phase (edge ids come from simulation/net/intersection.edg.xml).
APPROACH_EDGE_BY_DIRECTION: dict[Direction, str] = {
    Direction.EAST: "E2C",
    Direction.NORTH: "N2C",
    Direction.WEST: "W2C",
    Direction.SOUTH: "S2C",
}

# Fixed iteration order for building the flat 80-vector: matches Direction's
# integer values (EAST=0, NORTH=1, WEST=2, SOUTH=3), so
# global_cell_index = direction.value * GRID_CELLS_PER_APPROACH + cell_in_approach.
DIRECTION_ORDER: tuple[Direction, ...] = (
    Direction.EAST,
    Direction.NORTH,
    Direction.WEST,
    Direction.SOUTH,
)


def _cell_index_for_position(distance_to_stop_line_m: float) -> int:
    """Map a vehicle's distance from the stop line to a 0..19 cell index,
    clamped so vehicles beyond the modeled 100m segment still land in the
    last cell instead of being silently dropped."""
    if distance_to_stop_line_m < 0:
        distance_to_stop_line_m = 0.0
    idx = int(distance_to_stop_line_m // CELL_LENGTH_M)
    return min(idx, GRID_CELLS_PER_APPROACH - 1)


class GridEncoder:
    """Stateless encoder: call `.encode(traci_conn)` once per control step.

    `traci_conn` is whatever `TraciSession.traci` returns (the live traci
    module/connection) - this class never starts or owns a SUMO process
    itself, so it's trivially unit-testable against a real short-lived
    session and reusable from both the Gymnasium env (Stage 3) and any
    offline analysis script.
    """

    def __init__(self, edge_lengths_m: dict[str, float] | None = None) -> None:
        # netconvert shortens edges near a junction to fit its actual shape,
        # so the real lane length is rarely exactly APPROACH_LENGTH_M - query
        # it from TraCI (traci.lane.getLength) on first use and cache it,
        # rather than trusting the nominal 100m constant. `edge_lengths_m`
        # lets callers (tests, alternate networks) pre-seed/override entries.
        self._edge_length_cache: dict[str, float] = dict(edge_lengths_m or {})

    def _edge_length(self, traci_conn, edge_id: str) -> float:  # noqa: ANN001
        if edge_id not in self._edge_length_cache:
            self._edge_length_cache[edge_id] = traci_conn.lane.getLength(f"{edge_id}_0")
        return self._edge_length_cache[edge_id]

    def encode(self, traci_conn, step: int | None = None) -> IntersectionState:  # noqa: ANN001
        grid = [0.0] * GRID_CELLS_TOTAL
        vehicle_ids = traci_conn.vehicle.getIDList()

        for vehicle_id in vehicle_ids:
            edge_id = traci_conn.vehicle.getRoadID(vehicle_id)
            direction = self._direction_for_edge(edge_id)
            if direction is None:
                continue  # vehicle is inside the junction or on an outgoing edge

            edge_length = self._edge_length(traci_conn, edge_id)
            lane_position = traci_conn.vehicle.getLanePosition(vehicle_id)
            distance_to_stop_line = edge_length - lane_position

            cell_in_approach = _cell_index_for_position(distance_to_stop_line)
            global_index = direction.value * GRID_CELLS_PER_APPROACH + cell_in_approach
            grid[global_index] = 1.0

        resolved_step = step if step is not None else traci_conn.simulation.getTime()
        return IntersectionState(
            step=int(resolved_step),
            grid=grid,
            total_waiting_time_s=total_waiting_time(traci_conn, vehicle_ids),
        )

    @staticmethod
    def _direction_for_edge(edge_id: str) -> Direction | None:
        for direction, mapped_edge in APPROACH_EDGE_BY_DIRECTION.items():
            if mapped_edge == edge_id:
                return direction
        return None
