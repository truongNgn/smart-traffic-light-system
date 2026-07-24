"""Waiting-time primitives - the substrate for the Stage-3 reward engine
(rl/reward/), which computes r_t = t^T_{t-1} - t^T_t.

traci.vehicle.getWaitingTime(veh_id) already returns accumulated time since
that vehicle's speed last exceeded common.constants.STOPPED_SPEED_THRESHOLD_MPS
(SUMO's own default halting threshold is 0.1 m/s, i.e. exactly the paper's
threshold), so no reimplementation of the speed check is needed here - this
module just sums it across whichever vehicle population the caller cares about.
"""

from __future__ import annotations


def total_waiting_time(traci_conn, vehicle_ids: list[str] | None = None) -> float:  # noqa: ANN001
    """Sum of per-vehicle waiting time, in seconds.

    Pass `vehicle_ids` to scope the sum (e.g. only vehicles on the 4 approach
    edges); omit it to sum over every vehicle currently in the network.
    """
    ids = vehicle_ids if vehicle_ids is not None else traci_conn.vehicle.getIDList()
    return sum(traci_conn.vehicle.getWaitingTime(vid) for vid in ids)


def per_vehicle_waiting_time(traci_conn) -> dict[str, float]:  # noqa: ANN001
    """Same data, unaggregated - useful for debugging/dashboards (Engineer A)."""
    return {
        vid: traci_conn.vehicle.getWaitingTime(vid) for vid in traci_conn.vehicle.getIDList()
    }
