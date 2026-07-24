"""Epsilon-greedy DQN agent: policy network + target network + replay-buffer
training step. The full episode loop, epsilon decay schedule, and
experiment tracking belong to the Stage-4 training pipeline (rl/train/) -
this class is deliberately just the reusable core: pick an action, learn
from a batch.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn, optim

from common.constants import DQN_INPUT_SIZE, NUM_ACTIONS
from rl.agent.dqn import DQN
from rl.agent.replay_buffer import Transition


class DQNAgent:
    def __init__(
        self,
        input_size: int = DQN_INPUT_SIZE,
        num_actions: int = NUM_ACTIONS,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._rng = random.Random(seed)

        self.policy_net = DQN(input_size=input_size, output_size=num_actions).to(self.device)
        self.target_net = DQN(input_size=input_size, output_size=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection over the current Q-values."""
        if self._rng.random() < epsilon:
            return self._rng.randrange(self.num_actions)
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self, batch: list[Transition]) -> float:
        """One gradient step on a sampled replay batch. Returns the scalar loss."""
        states = torch.as_tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
