"""End-to-end DQN training loop over the SUMO Gymnasium env. Runnable
locally (small num_episodes, for pipeline verification) or on a GPU
notebook like Kaggle (large num_episodes, for a real policy) - the code
path is identical, only TrainingConfig's values differ. See
docs/training_on_kaggle.md for the Kaggle setup.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from tqdm.auto import tqdm

from common.logging import configure_logging, get_logger
from rl.agent.dqn_agent import DQNAgent
from rl.agent.replay_buffer import ReplayBuffer, Transition
from rl.env.traffic_env import SumoTrafficEnv
from rl.train.checkpoint import load_checkpoint, save_checkpoint
from rl.train.config import TrainingConfig

logger = get_logger(component="train")

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def epsilon_for_episode(episode_index: int, cfg: TrainingConfig) -> float:
    """Linear decay from epsilon_start to epsilon_end over
    epsilon_decay_episodes, then held flat at epsilon_end. episode_index is
    0-based (the first episode trains at epsilon_start)."""
    fraction = min(1.0, episode_index / cfg.epsilon_decay_episodes)
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def _resolve_backend(cfg: TrainingConfig) -> str:
    """libsumo is ~8x faster than traci for step-heavy workloads like
    training (no subprocess/socket round-trip per TraCI call - see
    simulation/traci_wrapper/session.py), so it's the default here. Fall
    back to traci with a warning rather than a hard failure if it isn't
    installed, so training still runs on a fresh checkout."""
    if cfg.backend != "libsumo":
        return cfg.backend
    try:
        import libsumo  # noqa: F401

        return "libsumo"
    except ImportError:
        logger.warning("train.libsumo_not_installed_falling_back_to_traci")
        return "traci"


def train(cfg: TrainingConfig) -> DQNAgent:
    configure_logging()

    backend = _resolve_backend(cfg)
    env = SumoTrafficEnv(
        sumocfg_path=cfg.sumocfg_path,
        use_gui=cfg.use_gui,
        seed=cfg.seed,
        episode_duration_s=cfg.episode_duration_s,
        green_duration_s=cfg.green_duration_s,
        backend=backend,
    )
    agent = DQNAgent(learning_rate=cfg.learning_rate, gamma=cfg.gamma, seed=cfg.seed)
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_episode = 0
    if cfg.resume_from:
        start_episode = load_checkpoint(cfg.resume_from, agent)
        logger.info("train.resumed", checkpoint=cfg.resume_from, from_episode=start_episode)

    use_mlflow = cfg.use_mlflow and _HAS_MLFLOW
    if cfg.use_mlflow and not _HAS_MLFLOW:
        logger.warning("train.mlflow_requested_but_not_installed")
    if use_mlflow:
        mlflow.set_experiment(cfg.mlflow_experiment_name)
        mlflow.start_run()
        mlflow.log_params(cfg.model_dump())

    best_episode_reward = -math.inf
    global_step = 0

    episode_range = range(start_episode + 1, cfg.num_episodes + 1)
    progress = tqdm(
        episode_range,
        desc="training",
        unit="ep",
        disable=not cfg.show_progress_bar,
        dynamic_ncols=True,
    )

    try:
        for episode in progress:
            epsilon = epsilon_for_episode(episode - 1, cfg)
            obs, _ = env.reset(seed=cfg.seed + episode)

            episode_reward = 0.0
            loss_sum, loss_count = 0.0, 0
            terminated = truncated = False
            final_waiting_time = 0.0

            while not (terminated or truncated):
                action = agent.act(obs, epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)
                buffer.push(Transition(obs, action, reward, next_obs, terminated))
                obs = next_obs
                episode_reward += reward
                final_waiting_time = info["total_waiting_time_s"]
                global_step += 1

                if len(buffer) >= cfg.min_replay_size and global_step % cfg.train_every_n_steps == 0:
                    batch = buffer.sample(cfg.batch_size)
                    loss_sum += agent.train_step(batch)
                    loss_count += 1

            if episode % cfg.target_sync_every_episodes == 0:
                agent.sync_target_network()

            avg_loss = loss_sum / loss_count if loss_count else 0.0
            progress.set_postfix(
                reward=f"{episode_reward:.0f}",
                eps=f"{epsilon:.2f}",
                loss=f"{avg_loss:.2f}",
                wait_s=f"{final_waiting_time:.0f}",
            )
            if episode % cfg.log_every_episodes == 0 or episode == cfg.num_episodes:
                logger.info(
                    "train.episode_done",
                    episode=episode,
                    reward=episode_reward,
                    epsilon=epsilon,
                    avg_loss=avg_loss,
                    final_waiting_time_s=final_waiting_time,
                    replay_size=len(buffer),
                )
            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "episode_reward": episode_reward,
                        "epsilon": epsilon,
                        "avg_loss": avg_loss,
                        "final_waiting_time_s": final_waiting_time,
                    },
                    step=episode,
                )

            if episode % cfg.checkpoint_every_episodes == 0 or episode == cfg.num_episodes:
                save_checkpoint(checkpoint_dir / f"dqn_episode_{episode}.pt", agent, episode)

            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                save_checkpoint(checkpoint_dir / "dqn_best.pt", agent, episode)

        save_checkpoint(checkpoint_dir / "dqn_final.pt", agent, cfg.num_episodes)
        logger.info(
            "train.complete",
            num_episodes=cfg.num_episodes,
            best_episode_reward=best_episode_reward,
            checkpoint_dir=str(checkpoint_dir),
        )
    finally:
        progress.close()
        env.close()
        if use_mlflow:
            mlflow.end_run()

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfg", dest="sumocfg_path", default=None)
    parser.add_argument("--gui", dest="use_gui", action="store_true", default=None)
    parser.add_argument("--episodes", dest="num_episodes", type=int, default=None)
    parser.add_argument("--episode-duration", dest="episode_duration_s", type=float, default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--resume", dest="resume_from", default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--mlflow", dest="use_mlflow", action="store_true", default=None)
    parser.add_argument(
        "--backend", dest="backend", choices=["libsumo", "traci"], default=None
    )
    parser.add_argument("--train-every-n-steps", dest="train_every_n_steps", type=int, default=None)
    parser.add_argument("--log-every-episodes", dest="log_every_episodes", type=int, default=None)
    parser.add_argument(
        "--no-progress-bar", dest="show_progress_bar", action="store_false", default=None
    )
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    cfg = TrainingConfig(**args)
    logger_ = get_logger(component="train")
    logger_.info("train.config", **cfg.model_dump())
    train(cfg)


if __name__ == "__main__":
    main()
