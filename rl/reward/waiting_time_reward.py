"""Reward engine: r_t = t^T_{t-1} - t^T_t (Sahal et al. 2023), where t^T_t is
the accumulated total waiting time of vehicles with speed < 0.1 m/s at time
t. A shrinking total waiting time (traffic clearing) yields positive reward;
a growing one (queues building) yields negative reward.

Delegates the actual accumulation to simulation/state/waiting_time.py,
which sums traci.vehicle.getWaitingTime() - SUMO's own halting-time metric,
already keyed to the paper's 0.1 m/s threshold (common.constants.
STOPPED_SPEED_THRESHOLD_MPS) - so this module is purely the t-1 vs t diff.
"""

from __future__ import annotations

from simulation.state.waiting_time import total_waiting_time


class WaitingTimeReward:
    """Stateful across an episode: call `reset()` once at episode start, then
    `step()` once per control step. Not thread-safe / not reusable across
    concurrent episodes - construct one instance per environment instance.
    """

    def __init__(self) -> None:
        self._previous_total: float = 0.0

    def reset(self, traci_conn) -> float:  # noqa: ANN001
        """Call right after the episode's first vehicles could plausibly
        exist (e.g. after the first simulationStep()). Returns t^T_0."""
        self._previous_total = total_waiting_time(traci_conn)
        return self._previous_total

    def step(self, traci_conn) -> float:  # noqa: ANN001
        """Returns r_t for the step just taken and advances internal state."""
        current_total = total_waiting_time(traci_conn)
        reward = self._previous_total - current_total
        self._previous_total = current_total
        return reward
