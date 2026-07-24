"""Golden tests for simulation/state/grid_encoder.py: place a vehicle at a
known, exact distance from the stop line via traci.vehicle.moveTo and assert
exactly which cell lights up. These are the tests referenced by
docs/state_encoding.md's worked example.

Real approach-edge length is queried at test time rather than assumed to be
the nominal 100m (netconvert trims edges to fit the junction's actual shape -
in this network they come out to ~89.6m, uniformly across all 4 directions).
Hardcoding the 100m constant here previously masked the exact bug this file
caught in GridEncoder itself; see docs/state_encoding.md.
"""

from __future__ import annotations

import pytest

from common.constants import CELL_LENGTH_M, GRID_CELLS_PER_APPROACH, GRID_CELLS_TOTAL, Direction
from simulation.state import APPROACH_EDGE_BY_DIRECTION, GridEncoder, total_waiting_time
from simulation.state.grid_encoder import _cell_index_for_position
from simulation.traci_wrapper import TraciSession
from tests.conftest import TEST_SUMOCFG, requires_network, requires_sumo


class TestCellIndexForPositionUnit:
    """Pure unit tests for the cell-math helper - no SUMO needed."""

    @pytest.mark.parametrize(
        "distance_m,expected_cell",
        [
            (0.0, 0),
            (0.1, 0),
            (4.99, 0),
            (5.0, 1),
            (8.0, 1),
            (95.0, 19),
            (99.99, 19),
            (250.0, 19),  # far beyond the modeled segment - must clamp, not raise/overflow
            (-1.0, 0),  # defensive: negative distance clamps to nearest cell
        ],
    )
    def test_cell_index_boundaries(self, distance_m: float, expected_cell: int) -> None:
        assert _cell_index_for_position(distance_m) == expected_cell

    def test_cell_length_matches_grid_constants(self) -> None:
        assert CELL_LENGTH_M * GRID_CELLS_PER_APPROACH == pytest.approx(100.0)


def _spawn_vehicle_at_distance(
    sim: TraciSession, vehicle_id: str, route_id: str, edge_id: str, distance_from_stop_line: float
) -> float:
    """Insert a vehicle and force it to an exact distance from the stop
    line, bypassing SUMO's normal insertion logic so the test is
    deterministic regardless of demand/traffic in the loaded .rou.xml.
    Returns the real edge length used, for the caller to compute expectations.
    """
    real_length = sim.traci.lane.getLength(f"{edge_id}_0")
    position = max(0.0, real_length - distance_from_stop_line)

    sim.traci.vehicle.add(vehicle_id, route_id, typeID="car", departSpeed="0")
    sim.step()  # vehicle must actually depart before moveTo will accept it
    sim.traci.vehicle.moveTo(vehicle_id, f"{edge_id}_0", position)
    sim.traci.simulationStep()  # let the forced position register
    return real_length


@requires_sumo
@requires_network
class TestGridEncoderGolden:
    def test_empty_intersection_has_all_zero_grid(self) -> None:
        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1, extra_args=["--no-warnings", "true"]) as sim:
            state = GridEncoder().encode(sim.traci, step=0)
        assert len(state.grid) == GRID_CELLS_TOTAL
        assert all(v in (0.0, 1.0) for v in state.grid)
        assert all(v == 0.0 for v in state.grid)

    @pytest.mark.parametrize(
        "direction,distance_from_stop_line_m",
        [
            # Cell 0 (nearest the stop line) isn't exercised here: SUMO treats a
            # vehicle as effectively arrived at the junction once it's within
            # roughly a vehicle-length-plus-minGap of the lane end, and moveTo's
            # position then gets absorbed onto the internal junction lane instead
            # of staying observable on the approach edge - see
            # TestCellIndexForPositionUnit for cell-0 coverage via pure math instead.
            (Direction.NORTH, 8.0),  # docs/state_encoding.md worked example -> cell 1
            (Direction.EAST, 8.0),
            (Direction.WEST, 8.0),
            (Direction.SOUTH, 8.0),
        ],
    )
    def test_vehicle_at_known_offset_lands_in_expected_cell(
        self, direction: Direction, distance_from_stop_line_m: float
    ) -> None:
        edge_id = APPROACH_EDGE_BY_DIRECTION[direction]
        route_id = _first_route_from(edge_id)

        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1, extra_args=["--no-warnings", "true"]) as sim:
            _spawn_vehicle_at_distance(sim, "golden_veh", route_id, edge_id, distance_from_stop_line_m)
            state = GridEncoder().encode(sim.traci, step=1)

        expected_cell_in_approach = _cell_index_for_position(distance_from_stop_line_m)
        expected_global_index = direction.value * GRID_CELLS_PER_APPROACH + expected_cell_in_approach
        occupied_cells = [i for i, v in enumerate(state.grid) if v == 1.0]

        assert occupied_cells == [expected_global_index], (
            f"expected only cell {expected_global_index} occupied "
            f"(direction={direction.name}, {distance_from_stop_line_m=}), got {occupied_cells}"
        )

    def test_vehicle_at_far_end_of_approach_lands_in_farthest_reachable_cell(self) -> None:
        # The real edge (~89.6m, trimmed by netconvert from the nominal
        # 100m) can't physically reach cell 19 (that needs >=95m clearance) -
        # the clamp-at-19 path is covered by the pure unit test above instead.
        # Requesting an absurdly large distance-from-stop-line clamps the
        # helper's position to 0.0 - i.e. the very start of the lane, as far
        # from the junction as this network allows - so the real farthest
        # reachable cell is whatever `real_length` (queried from TraCI, not
        # assumed) works out to.
        edge_id = APPROACH_EDGE_BY_DIRECTION[Direction.NORTH]
        route_id = _first_route_from(edge_id)

        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1, extra_args=["--no-warnings", "true"]) as sim:
            real_length = _spawn_vehicle_at_distance(
                sim, "golden_veh", route_id, edge_id, distance_from_stop_line=10_000.0
            )
            state = GridEncoder().encode(sim.traci, step=1)

        expected_cell_in_approach = _cell_index_for_position(real_length)
        expected_global_index = Direction.NORTH.value * GRID_CELLS_PER_APPROACH + expected_cell_in_approach
        occupied_cells = [i for i, v in enumerate(state.grid) if v == 1.0]
        assert occupied_cells == [expected_global_index]

    def test_vehicle_on_outgoing_edge_is_excluded_from_grid(self) -> None:
        # C2S is an outgoing edge (traffic leaving the junction southbound) -
        # the grid only represents approaching traffic, so a vehicle there
        # must not occupy any cell.
        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1, extra_args=["--no-warnings", "true"]) as sim:
            sim.traci.vehicle.add("outgoing_veh", "route_N_to_S", typeID="car", departSpeed="0")
            sim.step()
            sim.traci.vehicle.moveTo("outgoing_veh", "C2S_0", 10.0)
            sim.traci.simulationStep()
            state = GridEncoder().encode(sim.traci, step=1)

        assert all(v == 0.0 for v in state.grid)


@requires_sumo
@requires_network
class TestWaitingTime:
    def test_stationary_vehicle_accumulates_waiting_time(self) -> None:
        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1, extra_args=["--no-warnings", "true"]) as sim:
            sim.traci.vehicle.add("stuck_veh", "route_N_to_S", typeID="car", departSpeed="0")
            sim.step()
            sim.traci.vehicle.setSpeed("stuck_veh", 0.0)  # force-hold below the 0.1 m/s threshold
            for _ in range(10):
                sim.traci.simulationStep()
            waited = total_waiting_time(sim.traci, ["stuck_veh"])

        assert waited > 0.0

    def test_total_waiting_time_is_never_negative(self) -> None:
        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1) as sim:
            for _ in range(20):
                sim.step()
            waited = total_waiting_time(sim.traci)
        assert waited >= 0.0


def _first_route_from(edge_id: str) -> str:
    """Route ids follow route_<ORIGIN>_to_<DEST> (see generate_routes.py);
    any route starting on `edge_id` works since we override position with
    moveTo regardless of the route's destination."""
    origin = {v: k for k, v in {"N": "N2C", "S": "S2C", "E": "E2C", "W": "W2C"}.items()}[edge_id]
    destination = next(d for d in "NSEW" if d != origin)
    return f"route_{origin}_to_{destination}"
