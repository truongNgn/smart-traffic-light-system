"""Run a trained DQN checkpoint as the live control-facing agent.

This is the Stage-4 integration bridge: it uses Engineer B's trained model
and SUMO environment, then publishes Engineer A's bus contracts so the real
control service and dashboard observe the same decisions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import redis
import torch

from common.config import agent_runtime_settings, redis_settings, settings
from common.constants import PhaseAction as PhaseActionEnum
from common.logging import configure_logging, get_logger
from common.schemas.control import PhaseAction, ReasoningLog
from rl.agent.dqn_agent import DQNAgent
from rl.env.traffic_env import SumoTrafficEnv
from rl.train.checkpoint import load_checkpoint

logger = get_logger(component="agent_runtime")


def q_values_for(agent: DQNAgent, obs: np.ndarray) -> dict[str, float]:
    with torch.no_grad():
        state_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        values = agent.policy_net(state_t).squeeze(0).detach().cpu().tolist()
    return {PhaseActionEnum(i).name: float(value) for i, value in enumerate(values)}


def publish_decision(
    redis_client: redis.Redis,
    *,
    phase: PhaseActionEnum,
    q_values: dict[str, float],
    sim_time_s: float,
    total_waiting_time_s: float,
) -> None:
    timestamp_s = time.time()
    action = PhaseAction(target_phase=phase, timestamp_s=timestamp_s)
    reasoning = ReasoningLog(
        timestamp_s=timestamp_s,
        q_values=q_values,
        chosen_action=phase,
        exploration=False,
        sim_time_s=sim_time_s,
        total_waiting_time_s=total_waiting_time_s,
    )

    redis_client.xadd("agent_commands", {"data": action.model_dump_json()})
    redis_client.xadd("reasoning_logs", {"data": reasoning.model_dump_json()})


def run(
    *,
    checkpoint_path: str,
    sumocfg_path: str,
    episode_duration_s: float,
    seed: int,
    backend: str,
    use_gui: bool,
    decision_interval_s: float,
) -> None:
    configure_logging()

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    agent = DQNAgent()
    trained_episode = load_checkpoint(checkpoint, agent)
    redis_client = redis.Redis(
        host=redis_settings.host,
        port=redis_settings.port,
        decode_responses=True,
    )
    redis_client.ping()

    env = SumoTrafficEnv(
        sumocfg_path=sumocfg_path,
        episode_duration_s=episode_duration_s,
        seed=seed,
        backend=backend,
        use_gui=use_gui,
    )

    logger.info(
        "agent_runtime.started",
        checkpoint=str(checkpoint),
        trained_episode=trained_episode,
        sumocfg_path=sumocfg_path,
        episode_duration_s=episode_duration_s,
        backend=backend,
        decision_interval_s=decision_interval_s,
    )

    try:
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            q_values = q_values_for(agent, obs)
            action_index = max(q_values, key=q_values.get)
            phase = PhaseActionEnum[action_index]

            publish_decision(
                redis_client,
                phase=phase,
                q_values=q_values,
                sim_time_s=float(info["sim_time_s"]),
                total_waiting_time_s=float(info["total_waiting_time_s"]),
            )

            obs, reward, terminated, truncated, info = env.step(phase.value)
            logger.info(
                "agent_runtime.step",
                sim_time_s=info["sim_time_s"],
                phase=info["phase"],
                requested_phase=info["requested_phase"],
                reward=reward,
                total_waiting_time_s=info["total_waiting_time_s"],
            )
            if decision_interval_s > 0:
                time.sleep(decision_interval_s)
    finally:
        env.close()
        redis_client.close()

    logger.info("agent_runtime.finished")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=agent_runtime_settings.checkpoint_path)
    parser.add_argument("--sumocfg", default=settings.sumocfg_path)
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=agent_runtime_settings.episode_duration_s,
    )
    parser.add_argument("--seed", type=int, default=agent_runtime_settings.seed)
    parser.add_argument(
        "--backend",
        choices=["traci", "libsumo"],
        default=agent_runtime_settings.backend,
    )
    parser.add_argument(
        "--gui",
        dest="use_gui",
        action="store_true",
        default=agent_runtime_settings.use_gui,
    )
    parser.add_argument(
        "--decision-interval",
        type=float,
        default=agent_runtime_settings.decision_interval_s,
        help="Wall-clock seconds between published decisions; use 0 for fast smoke tests.",
    )
    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        sumocfg_path=args.sumocfg,
        episode_duration_s=args.episode_duration,
        seed=args.seed,
        backend=args.backend,
        use_gui=args.use_gui,
        decision_interval_s=args.decision_interval,
    )


if __name__ == "__main__":
    main()
