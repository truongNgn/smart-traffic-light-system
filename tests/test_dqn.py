"""Pure unit tests for the DQN architecture and agent - no SUMO needed.
Verifies the paper's exact shape (80 -> 5x400 -> 4) and that a training
step actually produces a gradient signal.
"""

from __future__ import annotations

import numpy as np
import torch

from common.constants import DQN_HIDDEN_LAYERS, DQN_HIDDEN_SIZE, DQN_INPUT_SIZE, DQN_OUTPUT_SIZE
from rl.agent.dqn import DQN
from rl.agent.dqn_agent import DQNAgent
from rl.agent.replay_buffer import ReplayBuffer, Transition


class TestDQNArchitecture:
    def test_forward_pass_produces_expected_output_shape(self) -> None:
        model = DQN()
        batch = torch.zeros((7, DQN_INPUT_SIZE))
        out = model(batch)
        assert out.shape == (7, DQN_OUTPUT_SIZE)

    def test_single_sample_forward_pass(self) -> None:
        model = DQN()
        single = torch.zeros((1, DQN_INPUT_SIZE))
        out = model(single)
        assert out.shape == (1, DQN_OUTPUT_SIZE)

    def test_hidden_layer_count_and_width_match_the_paper(self) -> None:
        model = DQN()
        linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        # DQN_HIDDEN_LAYERS hidden Linear layers + 1 output Linear layer.
        assert len(linear_layers) == DQN_HIDDEN_LAYERS + 1
        for layer in linear_layers[:-1]:
            assert layer.out_features == DQN_HIDDEN_SIZE
        assert linear_layers[-1].out_features == DQN_OUTPUT_SIZE
        assert linear_layers[0].in_features == DQN_INPUT_SIZE

    def test_custom_dimensions_are_respected(self) -> None:
        model = DQN(input_size=10, hidden_size=32, hidden_layers=2, output_size=3)
        out = model(torch.zeros((2, 10)))
        assert out.shape == (2, 3)


class TestReplayBuffer:
    def test_sample_raises_when_not_enough_transitions(self) -> None:
        buffer = ReplayBuffer(capacity=10)
        buffer.push(_dummy_transition())
        try:
            buffer.sample(5)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_capacity_evicts_oldest(self) -> None:
        buffer = ReplayBuffer(capacity=3)
        for i in range(5):
            buffer.push(_dummy_transition(action=i))
        assert len(buffer) == 3

    def test_sample_returns_requested_batch_size(self) -> None:
        buffer = ReplayBuffer(capacity=10, seed=0)
        for i in range(10):
            buffer.push(_dummy_transition(action=i))
        batch = buffer.sample(4)
        assert len(batch) == 4


class TestDQNAgent:
    def test_act_returns_valid_action_index(self) -> None:
        agent = DQNAgent(seed=0)
        state = np.zeros(DQN_INPUT_SIZE, dtype=np.float32)
        action = agent.act(state, epsilon=0.0)  # pure greedy
        assert 0 <= action < DQN_OUTPUT_SIZE

    def test_act_epsilon_one_is_pure_random_but_still_valid(self) -> None:
        agent = DQNAgent(seed=0)
        state = np.zeros(DQN_INPUT_SIZE, dtype=np.float32)
        actions = {agent.act(state, epsilon=1.0) for _ in range(50)}
        assert actions.issubset(set(range(DQN_OUTPUT_SIZE)))

    def test_train_step_reduces_loss_on_repeated_batch(self) -> None:
        agent = DQNAgent(seed=0, learning_rate=1e-2)
        batch = [_dummy_transition(action=i % DQN_OUTPUT_SIZE, reward=1.0) for i in range(16)]

        first_loss = agent.train_step(batch)
        for _ in range(20):
            loss = agent.train_step(batch)

        assert loss < first_loss

    def test_sync_target_network_matches_policy_weights(self) -> None:
        agent = DQNAgent(seed=0)
        # Nudge the policy net so it differs from a freshly-initialized target net.
        agent.train_step([_dummy_transition(action=0, reward=1.0) for _ in range(8)])
        agent.sync_target_network()
        for p_param, t_param in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
            assert torch.equal(p_param, t_param)


def _dummy_transition(action: int = 0, reward: float = 0.0) -> Transition:
    return Transition(
        state=np.zeros(DQN_INPUT_SIZE, dtype=np.float32),
        action=action,
        reward=reward,
        next_state=np.zeros(DQN_INPUT_SIZE, dtype=np.float32),
        done=False,
    )
