"""Golden tests for rl/reward/waiting_time_reward.py: r_t = t^T_{t-1} - t^T_t."""

from __future__ import annotations

from rl.reward.waiting_time_reward import WaitingTimeReward
from simulation.traci_wrapper import TraciSession
from tests.conftest import TEST_SUMOCFG, requires_network, requires_sumo


@requires_sumo
@requires_network
class TestWaitingTimeReward:
    def test_empty_network_reward_is_zero(self) -> None:
        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1) as sim:
            reward_engine = WaitingTimeReward()
            reward_engine.reset(sim.traci)
            sim.step()
            reward = reward_engine.step(sim.traci)
        assert reward == 0.0

    def test_growing_queue_yields_negative_reward(self) -> None:
        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1, extra_args=["--no-warnings", "true"]) as sim:
            reward_engine = WaitingTimeReward()

            sim.traci.vehicle.add("stuck_veh", "route_N_to_S", typeID="car", departSpeed="0")
            sim.step()
            sim.traci.vehicle.setSpeed("stuck_veh", 0.0)  # force-hold below 0.1 m/s

            reward_engine.reset(sim.traci)
            for _ in range(5):
                sim.traci.simulationStep()
            reward = reward_engine.step(sim.traci)

        # Waiting time only grew (one stopped vehicle, nothing cleared), so
        # t^T_t > t^T_{t-1} and reward = t^T_{t-1} - t^T_t < 0.
        assert reward < 0.0

    def test_reward_recovers_to_zero_once_vehicle_moves_again(self) -> None:
        with TraciSession(sumocfg_path=TEST_SUMOCFG, seed=1, extra_args=["--no-warnings", "true"]) as sim:
            reward_engine = WaitingTimeReward()

            sim.traci.vehicle.add("veh", "route_N_to_S", typeID="car", departSpeed="0")
            sim.step()
            sim.traci.vehicle.setSpeed("veh", 0.0)
            for _ in range(3):
                sim.traci.simulationStep()

            reward_engine.reset(sim.traci)
            # Release the forced stop; once the vehicle's own speed control
            # resumes, its waiting time stops accumulating and eventually resets.
            sim.traci.vehicle.setSpeed("veh", -1.0)
            rewards = []
            for _ in range(15):
                sim.traci.simulationStep()
                rewards.append(reward_engine.step(sim.traci))

        # Once moving again, waiting time should stop growing - i.e. we
        # should see at least one non-negative reward step after release.
        assert any(r >= 0.0 for r in rewards)
