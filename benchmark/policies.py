"""The two controllers being compared: a classic fixed-time cycle (the
paper's baseline) and the trained DQN agent, wired to the identical
interface so benchmark/run_episode.py can drive either one through the
same SumoTrafficEnv without knowing which it's holding.

Both run through SumoTrafficEnv, not raw TraCI - the fixed-time controller
does NOT reimplement phase timing itself. It only decides *which direction*
gets a green next; SumoTrafficEnv's `_apply_phase_transition` is still the
one and only place that inserts the yellow/all-red safety buffer. This
keeps the comparison apples-to-apples: both conditions experience the exact
same safety-buffer overhead, so any difference in the metrics is
attributable to the *decision policy*, not to one side getting a timing
advantage the other didn't.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from common.constants import Direction
from rl.agent.dqn_agent import DQNAgent


class Policy(Protocol):
    """Selects the next direction to grant a green phase to."""

    def select_action(self, obs: np.ndarray, sim_time_s: float) -> int: ...

    def reset(self) -> None: ...


class FixedTimePolicy:
    """Cycles through directions in a fixed order, each held green for
    `green_duration_s` before moving to the next - the standard fixed-time
    baseline referenced in Sahal et al. (2023) as the point of comparison
    for the DRL approach.

    Driven by absolute simulated time (`sim_time_s`, from SumoTrafficEnv's
    info dict), not by counting calls to select_action - a single call
    during a direction switch can itself advance several simulated seconds
    (the yellow+all-red buffer), so counting calls would drift the cycle
    out of sync with real time. Time-based cycling stays correct regardless.
    """

    def __init__(
        self,
        green_duration_s: float = 20.0,
        direction_order: tuple[Direction, ...] = (
            Direction.EAST,
            Direction.NORTH,
            Direction.WEST,
            Direction.SOUTH,
        ),
    ) -> None:
        self.green_duration_s = green_duration_s
        self.direction_order = direction_order
        self._cycle_length_s = green_duration_s * len(direction_order)

    def select_action(self, obs: np.ndarray, sim_time_s: float) -> int:  # noqa: ARG002
        position_in_cycle = sim_time_s % self._cycle_length_s
        phase_index = int(position_in_cycle // self.green_duration_s)
        phase_index = min(phase_index, len(self.direction_order) - 1)
        return self.direction_order[phase_index].value

    def reset(self) -> None:
        pass  # stateless - time-based, nothing to reset between episodes


class DQNPolicy:
    """Wraps a trained (or loading-in-progress) DQNAgent for greedy
    (epsilon=0 by default) evaluation - exploration noise has no place in a
    benchmark run."""

    def __init__(self, agent: DQNAgent, epsilon: float = 0.0) -> None:
        self.agent = agent
        self.epsilon = epsilon

    def select_action(self, obs: np.ndarray, sim_time_s: float) -> int:  # noqa: ARG002
        return self.agent.act(obs, self.epsilon)

    def reset(self) -> None:
        pass  # the network's weights are the only state, untouched by episodes
