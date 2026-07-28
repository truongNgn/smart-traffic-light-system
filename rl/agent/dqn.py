"""Deep Q-Network architecture: an 80-neuron input layer (one per grid cell,
see docs/state_encoding.md), 5 hidden layers of 400 neurons each with ReLU,
and one Q-value per common.constants.PhaseAction.
"""

from __future__ import annotations

import torch
from torch import nn

from common.constants import DQN_HIDDEN_LAYERS, DQN_HIDDEN_SIZE, DQN_INPUT_SIZE, DQN_OUTPUT_SIZE


class DQN(nn.Module):
    def __init__(
        self,
        input_size: int = DQN_INPUT_SIZE,
        hidden_size: int = DQN_HIDDEN_SIZE,
        hidden_layers: int = DQN_HIDDEN_LAYERS,
        output_size: int = DQN_OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        dims = [input_size] + [hidden_size] * hidden_layers
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_size) -> (batch, output_size) Q-values."""
        return self.net(x)
