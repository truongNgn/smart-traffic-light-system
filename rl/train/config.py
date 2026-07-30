"""All training knobs, env-driven (12-factor) so a Kaggle notebook can
override everything via env vars or CLI flags without touching code."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.constants import DEFAULT_SEED
from simulation.traci_wrapper.session import Backend


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAIN_", env_file=".env", extra="ignore")

    # Simulation
    sumocfg_path: str = Field(default="simulation/net/intersection.sumocfg")
    use_gui: bool = Field(default=False)
    seed: int = Field(default=DEFAULT_SEED)
    episode_duration_s: float = Field(default=3600.0)
    green_duration_s: float = Field(
        default=10.0,
        gt=0.0,
        description="Seconds to hold each selected green action; 10s matches the paper.",
    )
    max_red_time_s: float | None = Field(
        default=90.0,
        description="Safety guard: force a phase if it has been red longer than this.",
    )
    soft_red_time_s: float = Field(
        default=100.0,
        description="Adaptive guard: after this red time, force only if queue/waiting is high.",
    )
    hard_red_time_s: float = Field(
        default=150.0,
        description="Absolute max red time cap even when the starving phase has light demand.",
    )
    starving_queue_threshold: int = Field(
        default=8,
        ge=0,
        description="Queue threshold that activates the soft red-time guard.",
    )
    starving_wait_time_s: float = Field(
        default=300.0,
        ge=0.0,
        description="Per-phase waiting-time threshold that activates the soft red-time guard.",
    )
    backend: Backend = Field(
        default="libsumo",
        description="'libsumo' (default, ~8x faster - training never needs a GUI) or "
        "'traci' (subprocess+socket). train.py falls back to 'traci' with a "
        "warning if the libsumo package isn't installed.",
    )

    # Training loop
    num_episodes: int = Field(default=500, ge=1)
    replay_capacity: int = Field(default=50_000, ge=1)
    batch_size: int = Field(default=64, ge=1)
    min_replay_size: int = Field(
        default=1_000, ge=1, description="Steps of random experience before training starts."
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    train_every_n_steps: int = Field(
        default=4,
        ge=1,
        description="Run a gradient step every N environment steps instead of every "
        "single one - a standard DQN knob (e.g. Rainbow/DQN-Atari implementations "
        "default to 4) that cuts optimizer/backward-pass overhead roughly N-fold "
        "without changing what data ends up in the replay buffer.",
    )

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

    # Display
    log_every_episodes: int = Field(
        default=10,
        ge=1,
        description="Full structured JSON log line every N episodes (the live progress "
        "bar already shows every episode's stats - this just controls how much "
        "detailed JSON gets printed/persisted alongside it).",
    )
    show_progress_bar: bool = Field(default=True)

    # Optional experiment tracking - only activates if `mlflow` is
    # importable; never a hard dependency for training to run.
    use_mlflow: bool = Field(default=False)
    mlflow_experiment_name: str = Field(default="smart-traffic-dqn")
