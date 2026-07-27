"""Tests for rl/train/: pure-logic tests for epsilon decay and checkpoint
round-tripping, plus one end-to-end smoke test that runs a couple of tiny
episodes against real SUMO and checks a checkpoint file actually lands on
disk. This is the test that proves the full pipeline (env + agent + replay
buffer + checkpointing) works together before anyone spends GPU-hours on
Kaggle running the real thing.
"""

from __future__ import annotations

import pytest
import torch

from rl.agent.dqn_agent import DQNAgent
from rl.train.checkpoint import load_checkpoint, save_checkpoint
from rl.train.config import TrainingConfig
from rl.train.train import epsilon_for_episode, train
from tests.conftest import TEST_SUMOCFG, requires_network, requires_sumo


class TestEpsilonSchedule:
    def test_first_episode_uses_epsilon_start(self) -> None:
        cfg = TrainingConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_episodes=100)
        assert epsilon_for_episode(0, cfg) == 1.0

    def test_epsilon_decays_linearly_partway_through(self) -> None:
        cfg = TrainingConfig(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_episodes=100)
        assert epsilon_for_episode(50, cfg) == 0.5

    def test_epsilon_holds_at_end_value_past_decay_window(self) -> None:
        cfg = TrainingConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=100)
        assert epsilon_for_episode(500, cfg) == pytest.approx(0.05)

    def test_epsilon_never_exceeds_bounds(self) -> None:
        cfg = TrainingConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=10)
        for episode in range(0, 30):
            eps = epsilon_for_episode(episode, cfg)
            assert 0.05 <= eps <= 1.0


class TestCheckpointRoundTrip:
    def test_save_then_load_restores_identical_weights(self, tmp_path) -> None:  # noqa: ANN001
        agent = DQNAgent(seed=0)
        # Nudge weights away from their fresh-init state so a no-op load
        # couldn't accidentally pass this test.
        agent.train_step(_dummy_batch())

        checkpoint_path = tmp_path / "dqn_test.pt"
        save_checkpoint(checkpoint_path, agent, episode=7)

        restored_agent = DQNAgent(seed=1)  # different seed -> different init
        episode = load_checkpoint(checkpoint_path, restored_agent)

        assert episode == 7
        for p1, p2 in zip(agent.policy_net.parameters(), restored_agent.policy_net.parameters()):
            assert torch.equal(p1, p2)

    def test_checkpoint_creates_parent_directories(self, tmp_path) -> None:  # noqa: ANN001
        agent = DQNAgent(seed=0)
        nested_path = tmp_path / "a" / "b" / "c" / "dqn.pt"
        save_checkpoint(nested_path, agent, episode=1)
        assert nested_path.exists()


def _dummy_batch():
    import numpy as np

    from common.constants import DQN_INPUT_SIZE, DQN_OUTPUT_SIZE
    from rl.agent.replay_buffer import Transition

    return [
        Transition(
            state=np.zeros(DQN_INPUT_SIZE, dtype=np.float32),
            action=i % DQN_OUTPUT_SIZE,
            reward=1.0,
            next_state=np.zeros(DQN_INPUT_SIZE, dtype=np.float32),
            done=False,
        )
        for i in range(8)
    ]


@requires_sumo
@requires_network
class TestTrainingPipelineSmoke:
    def test_two_tiny_episodes_produce_a_checkpoint(self, tmp_path) -> None:  # noqa: ANN001
        cfg = TrainingConfig(
            sumocfg_path=str(TEST_SUMOCFG),
            num_episodes=2,
            episode_duration_s=10.0,  # a handful of control steps per episode
            replay_capacity=200,
            batch_size=4,
            min_replay_size=4,
            epsilon_decay_episodes=2,
            target_sync_every_episodes=1,
            checkpoint_dir=str(tmp_path),
            checkpoint_every_episodes=1,
            show_progress_bar=False,
        )
        agent = train(cfg)

        assert (tmp_path / "dqn_final.pt").exists()
        assert (tmp_path / "dqn_best.pt").exists()
        assert (tmp_path / "dqn_episode_1.pt").exists()
        assert (tmp_path / "dqn_episode_2.pt").exists()
        assert agent is not None

    def test_resume_continues_from_saved_episode(self, tmp_path) -> None:  # noqa: ANN001
        cfg = TrainingConfig(
            sumocfg_path=str(TEST_SUMOCFG),
            num_episodes=1,
            episode_duration_s=10.0,
            replay_capacity=200,
            batch_size=4,
            min_replay_size=4,
            checkpoint_dir=str(tmp_path),
            checkpoint_every_episodes=1,
            show_progress_bar=False,
        )
        train(cfg)

        resumed_cfg = cfg.model_copy(
            update={"num_episodes": 2, "resume_from": str(tmp_path / "dqn_final.pt")}
        )
        train(resumed_cfg)

        assert (tmp_path / "dqn_episode_2.pt").exists()
