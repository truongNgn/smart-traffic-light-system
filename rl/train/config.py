"""All training knobs, env-driven (12-factor) so a Kaggle notebook can
override everything via env vars or CLI flags without touching code."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.constants import DEFAULT_SEED


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAIN_", env_file=".env", extra="ignore")

    # Simulation
    sumocfg_path: str = Field(default="simulation/net/intersection.sumocfg")
    use_gui: bool = Field(default=False)
    seed: int = Field(default=DEFAULT_SEED)
    episode_duration_s: float = Field(default=3600.0)

    # Training loop
    num_episodes: int = Field(default=500, ge=1)
    replay_capacity: int = Field(default=50_000, ge=1)
    batch_size: int = Field(default=64, ge=1)
    min_replay_size: int = Field(
        default=1_000, ge=1, description="Steps of random experience before training starts."
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=1e-4, gt=0.0)

    # Epsilon-greedy exploration: linear decay from epsilon_start to
    # epsilon_end over epsilon_decay_episodes, then held at epsilon_end.
    epsilon_start: float = Field(default=1.0)
    epsilon_end: float = Field(default=0.05)
    epsilon_decay_episodes: int = Field(default=400, ge=1)

    target_sync_every_episodes: int = Field(default=10, ge=1)

    # Checkpointing
    checkpoint_dir: str = Field(default="checkpoints")
    checkpoint_every_episodes: int = Field(default=25, ge=1)
    resume_from: str | None = Field(default=None)

    # Optional experiment tracking - only activates if `mlflow` is
    # importable; never a hard dependency for training to run.
    use_mlflow: bool = Field(default=False)
    mlflow_experiment_name: str = Field(default="smart-traffic-dqn")
