"""Tests for benchmark/: the fixed-time baseline policy (pure logic) plus
integration tests running full episodes and the end-to-end compare()
pipeline against real SUMO. The compare() tests use a freshly-initialized
(untrained) DQNAgent checkpoint - they prove the *pipeline* is wired
correctly, not that a trained model beats the baseline (that only becomes
true after real training, e.g. on Kaggle).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.compare import compare
from benchmark.metrics import EpisodeMetrics
from benchmark.policies import DQNPolicy, FixedTimePolicy
from benchmark.run_episode import run_episode
from common.constants import Direction
from rl.agent.dqn_agent import DQNAgent
from rl.env.traffic_env import SumoTrafficEnv
from rl.train.checkpoint import save_checkpoint
from tests.conftest import SUMOCFG, requires_network, requires_sumo


class TestFixedTimePolicyUnit:
    def test_first_direction_at_time_zero(self) -> None:
        policy = FixedTimePolicy(green_duration_s=20.0)
        assert policy.select_action(obs=None, sim_time_s=0.0) == policy.direction_order[0].value

    def test_advances_to_second_direction_after_green_duration(self) -> None:
        policy = FixedTimePolicy(green_duration_s=20.0)
        assert policy.select_action(obs=None, sim_time_s=20.0) == policy.direction_order[1].value
        assert policy.select_action(obs=None, sim_time_s=39.9) == policy.direction_order[1].value

    def test_cycles_back_to_first_direction_after_full_cycle(self) -> None:
        policy = FixedTimePolicy(green_duration_s=20.0)
        cycle_length = 20.0 * len(policy.direction_order)
        assert policy.select_action(obs=None, sim_time_s=cycle_length) == policy.direction_order[0].value
        assert (
            policy.select_action(obs=None, sim_time_s=cycle_length * 3 + 5)
            == policy.direction_order[0].value
        )

    def test_default_order_covers_all_four_directions(self) -> None:
        policy = FixedTimePolicy(green_duration_s=10.0)
        assert set(policy.direction_order) == set(Direction)

    def test_custom_direction_order_is_respected(self) -> None:
        custom_order = (Direction.SOUTH, Direction.EAST, Direction.WEST, Direction.NORTH)
        policy = FixedTimePolicy(green_duration_s=15.0, direction_order=custom_order)
        assert policy.select_action(obs=None, sim_time_s=0.0) == Direction.SOUTH.value
        assert policy.select_action(obs=None, sim_time_s=15.0) == Direction.EAST.value


class TestEpisodeMetrics:
    def test_finalize_computes_means_and_maxes(self) -> None:
        metrics = EpisodeMetrics()
        metrics.record_step(waiting_time_s=10.0, queue_length=2, sim_time_s=1.0)
        metrics.record_step(waiting_time_s=20.0, queue_length=4, sim_time_s=2.0)
        metrics.record_step(waiting_time_s=30.0, queue_length=6, sim_time_s=3.0)
        metrics.finalize(arrived_vehicles=5)

        assert metrics.mean_waiting_time_s == pytest.approx(20.0)
        assert metrics.max_waiting_time_s == pytest.approx(30.0)
        assert metrics.mean_queue_length == pytest.approx(4.0)
        assert metrics.max_queue_length == 6
        assert metrics.final_waiting_time_s == pytest.approx(30.0)
        assert metrics.episode_duration_s == pytest.approx(3.0)
        assert metrics.arrived_vehicles == 5
        assert metrics.total_steps == 3

    def test_empty_metrics_finalize_without_error(self) -> None:
        metrics = EpisodeMetrics().finalize(arrived_vehicles=0)
        assert metrics.mean_waiting_time_s == 0.0
        assert metrics.total_steps == 0

    def test_to_dict_excludes_raw_sample_lists(self) -> None:
        metrics = EpisodeMetrics()
        metrics.record_step(waiting_time_s=1.0, queue_length=1, sim_time_s=1.0)
        d = metrics.finalize(arrived_vehicles=0).to_dict()
        assert "_waiting_time_samples" not in d
        assert "final_waiting_time_s" in d


@requires_sumo
@requires_network
class TestRunEpisode:
    def test_fixed_time_policy_completes_a_short_episode(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=SUMOCFG, seed=1, episode_duration_s=60)
        try:
            metrics = run_episode(env, FixedTimePolicy(green_duration_s=15.0), seed=1)
        finally:
            env.close()
        assert metrics.total_steps > 0
        assert metrics.episode_duration_s > 0

    def test_dqn_policy_completes_a_short_episode(self) -> None:
        env = SumoTrafficEnv(sumocfg_path=SUMOCFG, seed=1, episode_duration_s=60)
        try:
            agent = DQNAgent(seed=0)
            metrics = run_episode(env, DQNPolicy(agent, epsilon=0.0), seed=1)
        finally:
            env.close()
        assert metrics.total_steps > 0


@requires_sumo
@requires_network
class TestComparePipeline:
    def test_compare_runs_end_to_end_and_returns_expected_shape(self, tmp_path: Path) -> None:
        # An untrained checkpoint is enough to prove the pipeline works -
        # it won't beat the baseline, that's expected and fine here.
        agent = DQNAgent(seed=0)
        checkpoint_path = tmp_path / "untrained.pt"
        save_checkpoint(checkpoint_path, agent, episode=0)

        report = compare(
            checkpoint_path=str(checkpoint_path),
            sumocfg_path=str(SUMOCFG),
            episode_duration_s=60.0,
            seeds=[1],
            fixed_green_duration_s=15.0,
        )

        assert "aggregated" in report
        assert set(report["aggregated"].keys()) == {"fixed_time", "dqn"}
        assert "improvement_pct" in report
        for key in ("final_waiting_time_s", "mean_waiting_time_s", "mean_queue_length", "arrived_vehicles"):
            assert key in report["improvement_pct"]
        assert report["per_seed"]["fixed_time"][0]["total_steps"] > 0
        assert report["per_seed"]["dqn"][0]["total_steps"] > 0

    def test_compare_multiple_seeds_produces_one_run_each(self, tmp_path: Path) -> None:
        agent = DQNAgent(seed=0)
        checkpoint_path = tmp_path / "untrained.pt"
        save_checkpoint(checkpoint_path, agent, episode=0)

        report = compare(
            checkpoint_path=str(checkpoint_path),
            sumocfg_path=str(SUMOCFG),
            episode_duration_s=30.0,
            seeds=[1, 2],
            fixed_green_duration_s=15.0,
        )

        assert len(report["per_seed"]["fixed_time"]) == 2
        assert len(report["per_seed"]["dqn"]) == 2
