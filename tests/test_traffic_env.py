"""Integration tests for rl/env/traffic_env.py: the Gymnasium env wrapping
SUMO/TraCI. Focused on the properties that matter for RL correctness and
deployment safety - observation/action space shape, and that every phase
change actually goes through the yellow + all-red buffer.
"""

from __future__ import annotations

import numpy as np

from common.constants import GRID_CELLS_TOTAL, NUM_ACTIONS
from rl.env.traffic_env import SumoTrafficEnv
from tests.conftest import SUMOCFG, TEST_SUMOCFG, requires_libsumo, requires_network, requires_sumo


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
            assert "sim_time_s" in info
            assert "arrived_vehicles" in info
        finally:
            env.close()

    def test_arrived_vehicles_is_cumulative_and_counts_transitions(self) -> None:
        # Uses the real (with-demand) sumocfg, not the routes-only test one,
        # so vehicles actually complete routes and get counted as arrived.
        env = SumoTrafficEnv(sumocfg_path=SUMOCFG, seed=1, episode_duration_s=300)
        try:
            obs, info = env.reset()
            assert info["arrived_vehicles"] == 0

            total_arrived = 0
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < 60:
                # Alternate actions every step to force a phase transition
                # (yellow+all-red+green) on every single call - the scenario
                # that would silently drop arrivals without _step_and_track.
                action = steps % NUM_ACTIONS
                obs, reward, terminated, truncated, info = env.step(action)
                assert info["arrived_vehicles"] >= total_arrived
                total_arrived = info["arrived_vehicles"]
                steps += 1

            assert total_arrived > 0, "No vehicles arrived - demand or routing may be broken."
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
            initial_phase_action = int(env._current_phase.value)
            step_before = env._sim.traci.simulation.getTime()
            obs, reward, terminated, truncated, info = env.step(initial_phase_action)
            step_after = env._sim.traci.simulation.getTime()
            # Staying on the same direction should hold the selected green
            # for the configured control interval, without yellow/all-red.
            assert step_after - step_before == env._green_steps * env.step_length_s
            assert obs.shape == (GRID_CELLS_TOTAL,)
        finally:
            env.close()

    def test_switching_direction_advances_through_full_safety_buffer(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1)
        try:
            env.reset()
            current_action = int(env._current_phase.value)
            other_action = (current_action + 1) % NUM_ACTIONS

            step_before = env._sim.traci.simulation.getTime()
            env.step(other_action)
            step_after = env._sim.traci.simulation.getTime()

            expected_steps = (
                env._yellow_steps + env._all_red_steps + env._green_steps
            ) * env.step_length_s
            assert step_after - step_before == expected_steps
            assert int(env._current_phase.value) == other_action
        finally:
            env.close()

    def test_max_red_time_guard_forces_starving_phase(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1, max_red_time_s=5.0)
        try:
            env.reset()
            initial_phase_action = int(env._current_phase.value)
            env.step(initial_phase_action)

            obs, reward, terminated, truncated, info = env.step(initial_phase_action)

            assert info["action_forced_by_guard"] is True
            assert int(env._current_phase.value) != initial_phase_action
            assert info["phase"] != info["requested_phase"]
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
            env.step(int(env._current_phase.value))
            step_before_second_reset = env._sim.traci.simulation.getTime()
            assert step_before_second_reset > 0

            env.reset()
            assert env._sim.traci.simulation.getTime() < step_before_second_reset
        finally:
            env.close()


@requires_sumo
@requires_network
@requires_libsumo
class TestSumoTrafficEnvLibsumoBackend:
    """rl/train/train.py defaults to backend='libsumo' for speed - proves
    the env produces the same shape of observation and honors the same
    safety buffer under that backend, not just the default 'traci' one."""

    def test_reset_and_step_produce_valid_observations(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1, backend="libsumo")
        try:
            obs, _ = env.reset()
            assert obs.shape == (GRID_CELLS_TOTAL,)
            other_action = (int(env._current_phase.value) + 1) % NUM_ACTIONS
            obs, reward, terminated, truncated, info = env.step(other_action)
            assert obs.shape == (GRID_CELLS_TOTAL,)
        finally:
            env.close()

    def test_switching_direction_still_honors_safety_buffer(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=TEST_SUMOCFG, seed=1, backend="libsumo")
        try:
            env.reset()
            current_action = int(env._current_phase.value)
            other_action = (current_action + 1) % NUM_ACTIONS

            step_before = env._sim.traci.simulation.getTime()
            env.step(other_action)
            step_after = env._sim.traci.simulation.getTime()

            expected_steps = (
                env._yellow_steps + env._all_red_steps + env._green_steps
            ) * env.step_length_s
            assert step_after - step_before == expected_steps
        finally:
            env.close()
