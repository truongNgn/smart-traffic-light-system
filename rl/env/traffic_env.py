"""Gymnasium environment wrapping SUMO + TraCI. Ties together everything
built in Stages 1-3: TraciSession (connection lifecycle), GridEncoder
(observation), TlsController (phase-switch safety), and WaitingTimeReward
(reward).

Critically, `_apply_phase_transition` is the *only* place phase changes
happen, and it always runs the mandatory yellow + all-red buffer
(common.constants.YELLOW_DURATION_S / ALL_RED_DURATION_S) before granting a
new green - the same constants and the same TlsController class that the
Stage-4 production control executor will use. An agent trained against a
buffer-free or differently-timed env would learn an unsafe policy once
deployed against the real executor.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.constants import (
    ALL_RED_DURATION_S,
    DEFAULT_SEED,
    DEFAULT_STEP_LENGTH_S,
    GREEN_DURATION_S,
    GRID_CELLS_TOTAL,
    NUM_ACTIONS,
    YELLOW_DURATION_S,
    PhaseAction,
)
from simulation.state.grid_encoder import GridEncoder
from simulation.traci_wrapper import TlsController, TraciSession
from rl.reward.waiting_time_reward import WaitingTimeReward


class SumoTrafficEnv(gym.Env):
    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        sumocfg_path: str | Path,
        *,
        tls_id: str = "C",
        use_gui: bool = False,
        seed: int = DEFAULT_SEED,
        step_length_s: float = DEFAULT_STEP_LENGTH_S,
        episode_duration_s: float = 3600.0,
        green_duration_s: float = GREEN_DURATION_S,
        max_red_time_s: float | None = 90.0,
        soft_red_time_s: float = 100.0,
        hard_red_time_s: float = 150.0,
        starving_queue_threshold: int = 8,
        starving_wait_time_s: float = 300.0,
        initial_phase: PhaseAction = PhaseAction.EAST_WEST,
        backend: str = "traci",
        sumo_extra_args: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.sumocfg_path = sumocfg_path
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.seed_value = seed
        self.step_length_s = step_length_s
        self.episode_duration_s = episode_duration_s
        self.green_duration_s = green_duration_s
        self.max_red_time_s = max_red_time_s
        self.soft_red_time_s = soft_red_time_s
        self.hard_red_time_s = hard_red_time_s
        self.starving_queue_threshold = starving_queue_threshold
        self.starving_wait_time_s = starving_wait_time_s
        self.initial_phase = initial_phase
        self.backend = backend
        self.sumo_extra_args = sumo_extra_args or []

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(GRID_CELLS_TOTAL,), dtype=np.float32
        )

        self._green_steps = max(1, round(green_duration_s / step_length_s))
        self._yellow_steps = max(1, round(YELLOW_DURATION_S / step_length_s))
        self._all_red_steps = max(1, round(ALL_RED_DURATION_S / step_length_s))

        self._sim: TraciSession | None = None
        self._tls: TlsController | None = None
        self._encoder = GridEncoder()
        self._reward_engine = WaitingTimeReward()
        self._current_phase: PhaseAction | None = None
        self._red_time_by_phase: dict[PhaseAction, float] = {phase: 0.0 for phase in PhaseAction}
        self._last_action_was_forced = False
        self._last_guard_reason: str | None = None
        self._cumulative_arrived = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # noqa: ANN001
        super().reset(seed=seed)
        if self._sim is not None:
            self._sim.close()

        episode_seed = seed if seed is not None else self.seed_value
        self._sim = TraciSession(
            sumocfg_path=self.sumocfg_path,
            use_gui=self.use_gui,
            seed=episode_seed,
            step_length_s=self.step_length_s,
            backend=self.backend,
            extra_args=self.sumo_extra_args,
        ).start()
        self._tls = TlsController(self._sim.traci, self.tls_id)
        self._cumulative_arrived = 0
        self._red_time_by_phase = {phase: 0.0 for phase in PhaseAction}
        self._last_action_was_forced = False
        self._last_guard_reason = None

        # First green is granted directly - there's no prior direction to
        # transition away from, so no yellow/all-red buffer applies yet.
        self._sim.traci.trafficlight.setRedYellowGreenState(
            self.tls_id, self._tls.green_state(self.initial_phase)
        )
        self._current_phase = self.initial_phase
        self._step_and_track(green_phase=self.initial_phase)

        self._reward_engine.reset(self._sim.traci)
        state = self._encoder.encode(self._sim.traci)
        obs = np.asarray(state.grid, dtype=np.float32)
        info = {
            "total_waiting_time_s": state.total_waiting_time_s,
            "sim_time_s": float(state.step),
            "arrived_vehicles": self._cumulative_arrived,
        }
        return obs, info

    def step(self, action: int):
        if self._sim is None:
            raise RuntimeError("Call reset() before step().")

        requested_phase = PhaseAction(action)
        phase = self._phase_after_safety_guard(requested_phase)
        if phase != self._current_phase:
            self._apply_phase_transition(phase)
        else:
            self._step_and_track(self._green_steps, green_phase=phase)
        self._current_phase = phase

        state = self._encoder.encode(self._sim.traci)
        reward = self._reward_engine.step(self._sim.traci)

        terminated = self._sim.traci.simulation.getMinExpectedNumber() <= 0
        truncated = state.step >= self.episode_duration_s

        obs = np.asarray(state.grid, dtype=np.float32)
        info = {
            "total_waiting_time_s": state.total_waiting_time_s,
            "sim_time_s": float(state.step),
            "arrived_vehicles": self._cumulative_arrived,
            "phase": phase.name,
            "requested_phase": requested_phase.name,
            "action_forced_by_guard": self._last_action_was_forced,
            "guard_reason": self._last_guard_reason,
            "red_time_s": {phase.name: red_time for phase, red_time in self._red_time_by_phase.items()},
        }
        return obs, reward, terminated, truncated, info

    def _phase_after_safety_guard(self, requested_phase: PhaseAction) -> PhaseAction:
        self._last_action_was_forced = False
        self._last_guard_reason = None
        if self.max_red_time_s is None and self.hard_red_time_s <= 0:
            return requested_phase

        forced_phase, reason = self._forced_phase_and_reason()
        if forced_phase is None:
            return requested_phase

        self._last_action_was_forced = forced_phase != requested_phase
        self._last_guard_reason = reason
        return forced_phase

    def _forced_phase_and_reason(self) -> tuple[PhaseAction | None, str | None]:
        candidates = [
            (phase, red_time_s)
            for phase, red_time_s in self._red_time_by_phase.items()
            if phase != self._current_phase
        ]
        if not candidates:
            return None, None

        hard_limit = self.hard_red_time_s
        if self.max_red_time_s is not None:
            hard_limit = max(hard_limit, self.max_red_time_s)

        for phase, red_time_s in candidates:
            if hard_limit > 0 and red_time_s >= hard_limit:
                return phase, "hard_red_time"

        for phase, red_time_s in candidates:
            if red_time_s < self.soft_red_time_s:
                continue
            queue = self._phase_queue_length(phase)
            waiting = self._phase_waiting_time(phase)
            if queue >= self.starving_queue_threshold:
                return phase, "soft_red_queue"
            if waiting >= self.starving_wait_time_s:
                return phase, "soft_red_wait"

        return None, None

    def _apply_phase_transition(self, new_phase: PhaseAction) -> None:
        assert self._sim is not None and self._tls is not None
        if self._current_phase is not None:
            self._sim.traci.trafficlight.setRedYellowGreenState(
                self.tls_id, self._tls.yellow_state(self._current_phase)
            )
            self._step_and_track(self._yellow_steps)

            self._sim.traci.trafficlight.setRedYellowGreenState(self.tls_id, self._tls.all_red_state())
            self._step_and_track(self._all_red_steps)

        self._sim.traci.trafficlight.setRedYellowGreenState(
            self.tls_id, self._tls.green_state(new_phase)
        )
        self._step_and_track(self._green_steps, green_phase=new_phase)

    def _step_and_track(self, n: int = 1, *, green_phase: PhaseAction | None = None) -> None:
        """Advance n simulation steps, accumulating arrived-vehicle counts
        along the way. Must step one-at-a-time (not TraciSession.step(n) in
        one call) so no arrivals during a multi-step yellow/all-red buffer
        are missed - traci.simulation.getArrivedNumber() only reports
        arrivals since the *last* simulationStep()."""
        assert self._sim is not None
        for _ in range(n):
            self._sim.step(1)
            self._update_red_times(green_phase)
            self._cumulative_arrived += self._sim.traci.simulation.getArrivedNumber()

    def _update_red_times(self, green_phase: PhaseAction | None) -> None:
        for phase in PhaseAction:
            if phase == green_phase:
                self._red_time_by_phase[phase] = 0.0
            else:
                self._red_time_by_phase[phase] += self.step_length_s

    def _phase_queue_length(self, phase: PhaseAction) -> int:
        assert self._sim is not None
        edges = self._phase_edges(phase)
        return sum(
            1
            for vehicle_id in self._sim.traci.vehicle.getIDList()
            if self._sim.traci.vehicle.getRoadID(vehicle_id) in edges
        )

    def _phase_waiting_time(self, phase: PhaseAction) -> float:
        assert self._sim is not None
        edges = self._phase_edges(phase)
        return sum(
            self._sim.traci.vehicle.getWaitingTime(vehicle_id)
            for vehicle_id in self._sim.traci.vehicle.getIDList()
            if self._sim.traci.vehicle.getRoadID(vehicle_id) in edges
        )

    @staticmethod
    def _phase_edges(phase: PhaseAction) -> set[str]:
        from common.constants import PHASE_DIRECTIONS
        from simulation.state.grid_encoder import APPROACH_EDGE_BY_DIRECTION

        return {APPROACH_EDGE_BY_DIRECTION[direction] for direction in PHASE_DIRECTIONS[phase]}

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
