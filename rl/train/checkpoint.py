"""Checkpoint save/load - a single .pt file per checkpoint containing the
policy net, target net, optimizer state, and episode count, so training can
resume exactly where it left off (e.g. across a Kaggle session time limit).
"""

from __future__ import annotations

from pathlib import Path

import torch

from rl.agent.dqn_agent import DQNAgent


def save_checkpoint(path: str | Path, agent: DQNAgent, episode: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, agent: DQNAgent) -> int:
    """Loads weights/optimizer state into `agent` in place. Returns the
    episode number the checkpoint was saved at, so the training loop can
    resume from episode + 1."""
    checkpoint = torch.load(Path(path), map_location=agent.device)
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["episode"])
