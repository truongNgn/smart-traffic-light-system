"""Integration tests for rl/env/traffic_env.py: the Gymnasium env wrapping
SUMO/TraCI. Focused on the properties that matter for RL correctness and
deployment safety - observation/action space shape, and that every phase
change actually goes through the yellow + all-red buffer.
"""

from __future__ import annotations

import numpy as np

from common.constants import GRID_CELLS_TOTAL, NUM_ACTIONS
from rl.env.traffic_env import SumoTrafficEnv
from tests.conftest import TEST_SUMOCFG, requires_network, requires_sumo


@requires_sumo
@requires_network
class TestSumoTrafficEnv:
    def test_reset_returns_valid_observation(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1)
        try:
            obs, info = env.reset()
            assert obs.shape == (GRID_CELLS_TOTAL,)
            assert obs.dtype == np.float32
            assert set(np.unique(obs)).issubset({0.0, 1.0})
            assert "total_waiting_time_s" in info
        finally:
            env.close()

    def test_action_space_has_4_discrete_actions(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1)
        try:
            assert env.action_space.n == NUM_ACTIONS
        finally:
            env.close()

    def test_step_same_direction_does_not_trigger_safety_buffer(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1)
        try:
            env.reset()
            initial_direction_action = int(env._current_direction.value)
            step_before = env._sim.traci.simulation.getTime()
            obs, reward, terminated, truncated, info = env.step(initial_direction_action)
            step_after = env._sim.traci.simulation.getTime()
            # Staying on the same direction should advance exactly one
            # control step (step_length_s), not yellow_steps + all_red_steps + 1.
            assert step_after - step_before == env.step_length_s
            assert obs.shape == (GRID_CELLS_TOTAL,)
        finally:
            env.close()

    def test_switching_direction_advances_through_full_safety_buffer(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1)
        try:
            env.reset()
            current_action = int(env._current_direction.value)
            other_action = (current_action + 1) % NUM_ACTIONS

            step_before = env._sim.traci.simulation.getTime()
            env.step(other_action)
            step_after = env._sim.traci.simulation.getTime()

            expected_steps = (
                env._yellow_steps + env._all_red_steps + 1
            ) * env.step_length_s
            assert step_after - step_before == expected_steps
            assert int(env._current_direction.value) == other_action
        finally:
            env.close()

    def test_observation_space_matches_returned_shape(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1)
        try:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)
        finally:
            env.close()

    def test_reset_is_idempotent_and_restarts_sumo(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1)
        try:
            env.reset()
            env.step(int(env._current_direction.value))
            step_before_second_reset = env._sim.traci.simulation.getTime()
            assert step_before_second_reset > 0

            env.reset()
            assert env._sim.traci.simulation.getTime() < step_before_second_reset
        finally:
            env.close()
