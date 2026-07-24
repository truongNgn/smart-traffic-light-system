"""Fixed-size experience replay buffer for off-policy DQN training."""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only has {len(self._buffer)} transitions."
            )
        return self._rng.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
