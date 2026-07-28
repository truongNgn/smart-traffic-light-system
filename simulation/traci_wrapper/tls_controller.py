"""Builds the traffic-light signal-state strings SUMO's TraCI expects
(e.g. "GGGgrrrrGGGgrrrr") for each of the 4 discrete actions, plus their
yellow and all-red counterparts.

This is the single source of truth both the RL environment (rl/env/) and
the production control executor (Stage 4, Engineer A's control/ package)
must use - defining the phase-to-signal-string mapping twice is exactly how
a policy trained in simulation ends up unsafe once wired to a real
executor. See common/constants.py's module docstring and
docs/state_encoding.md for the related "single source of truth" reasoning
on timing constants.
"""

from __future__ import annotations

from common.constants import PHASE_DIRECTIONS, Direction, PhaseAction
from simulation.state.grid_encoder import APPROACH_EDGE_BY_DIRECTION


def _direction_for_incoming_lane(lane_id: str) -> Direction | None:
    edge_id = lane_id.rsplit("_", 1)[0]
    for direction, mapped_edge in APPROACH_EDGE_BY_DIRECTION.items():
        if mapped_edge == edge_id:
            return direction
    return None


class TlsController:
    """Queries the network's actual signal-index layout once (via
    traci.trafficlight.getControlledLinks) instead of assuming a fixed
    string layout - netconvert's auto-generated index order isn't part of
    any documented contract, so this must stay derived, not hardcoded.
    """

    def __init__(self, traci_conn, tls_id: str = "C") -> None:  # noqa: ANN001
        self.tls_id = tls_id
        self._num_signals = len(traci_conn.trafficlight.getRedYellowGreenState(tls_id))
        self._indices_by_direction: dict[Direction, list[int]] = {d: [] for d in Direction}

        controlled_links = traci_conn.trafficlight.getControlledLinks(tls_id)
        for signal_index, link_list in enumerate(controlled_links):
            if not link_list:
                continue  # unused signal index (can happen on some networks)
            incoming_lane = link_list[0][0]
            direction = _direction_for_incoming_lane(incoming_lane)
            if direction is not None:
                self._indices_by_direction[direction].append(signal_index)

        for direction, indices in self._indices_by_direction.items():
            if not indices:
                raise ValueError(
                    f"No TLS signal indices found for {direction.name} at junction "
                    f"{tls_id!r} - check APPROACH_EDGE_BY_DIRECTION against the network."
                )

    def green_state(self, phase: PhaseAction) -> str:
        """Every signal for both opposite approaches in `phase` is green."""
        return self._state_with_phase(phase, "G")

    def yellow_state(self, phase: PhaseAction) -> str:
        """Every signal for the losing `phase` is yellow; everything else is red -
        used for the mandatory yellow buffer when `phase` is losing the
        green (common.constants.YELLOW_DURATION_S)."""
        return self._state_with_phase(phase, "y")

    def all_red_state(self) -> str:
        return "r" * self._num_signals

    def _state_with_phase(self, phase: PhaseAction, char: str) -> str:
        chars = ["r"] * self._num_signals
        for direction in PHASE_DIRECTIONS[phase]:
            for i in self._indices_by_direction[direction]:
                chars[i] = char
        return "".join(chars)
