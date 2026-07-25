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
    GRID_CELLS_TOTAL,
    NUM_ACTIONS,
    YELLOW_DURATION_S,
    Direction,
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
        initial_direction: Direction = Direction.EAST,
        backend: str = "traci",
    ) -> None:
        super().__init__()
        self.sumocfg_path = sumocfg_path
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.seed_value = seed
        self.step_length_s = step_length_s
        self.episode_duration_s = episode_duration_s
        self.initial_direction = initial_direction
        self.backend = backend

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(GRID_CELLS_TOTAL,), dtype=np.float32
        )

        self._yellow_steps = max(1, round(YELLOW_DURATION_S / step_length_s))
        self._all_red_steps = max(1, round(ALL_RED_DURATION_S / step_length_s))

        self._sim: TraciSession | None = None
        self._tls: TlsController | None = None
        self._encoder = GridEncoder()
        self._reward_engine = WaitingTimeReward()
        self._current_direction: Direction | None = None

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
        ).start()
        self._tls = TlsController(self._sim.traci, self.tls_id)

        # First green is granted directly - there's no prior direction to
        # transition away from, so no yellow/all-red buffer applies yet.
        self._sim.traci.trafficlight.setRedYellowGreenState(
            self.tls_id, self._tls.green_state(self.initial_direction)
        )
        self._current_direction = self.initial_direction
        self._sim.step()

        self._reward_engine.reset(self._sim.traci)
        state = self._encoder.encode(self._sim.traci)
        obs = np.asarray(state.grid, dtype=np.float32)
        info = {"total_waiting_time_s": state.total_waiting_time_s}
        return obs, info

    def step(self, action: int):
        if self._sim is None:
            raise RuntimeError("Call reset() before step().")

        direction = Direction(action)
        if direction != self._current_direction:
            self._apply_phase_transition(direction)
        else:
            self._sim.step()
        self._current_direction = direction

        state = self._encoder.encode(self._sim.traci)
        reward = self._reward_engine.step(self._sim.traci)

        terminated = self._sim.traci.simulation.getMinExpectedNumber() <= 0
        truncated = state.step >= self.episode_duration_s

        obs = np.asarray(state.grid, dtype=np.float32)
        info = {"total_waiting_time_s": state.total_waiting_time_s, "direction": direction.name}
        return obs, reward, terminated, truncated, info

    def _apply_phase_transition(self, new_direction: Direction) -> None:
        assert self._sim is not None and self._tls is not None
        if self._current_direction is not None:
            self._sim.traci.trafficlight.setRedYellowGreenState(
                self.tls_id, self._tls.yellow_state(self._current_direction)
            )
            self._sim.step(self._yellow_steps)

            self._sim.traci.trafficlight.setRedYellowGreenState(self.tls_id, self._tls.all_red_state())
            self._sim.step(self._all_red_steps)

        self._sim.traci.trafficlight.setRedYellowGreenState(
            self.tls_id, self._tls.green_state(new_direction)
        )
        self._sim.step()

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
