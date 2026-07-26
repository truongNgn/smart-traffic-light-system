"""Run a trained DQN policy in sumo-gui for visual inspection.

Example:
    python -m benchmark.run_dqn_gui --checkpoint checkpoints/dqn_best.pt
"""

from __future__ import annotations

import argparse

from common.constants import Direction
from common.logging import configure_logging
from benchmark.policies import DQNPolicy
from rl.agent.dqn_agent import DQNAgent
from rl.env.traffic_env import SumoTrafficEnv
from rl.train.checkpoint import load_checkpoint


def run_gui(
    *,
    checkpoint_path: str,
    sumocfg_path: str,
    episode_duration_s: float,
    seed: int,
    delay_ms: int,
    log_every_s: float,
) -> None:
    configure_logging()
    agent = DQNAgent()
    episode = load_checkpoint(checkpoint_path, agent)
    policy = DQNPolicy(agent, epsilon=0.0)
    env = SumoTrafficEnv(
        sumocfg_path=sumocfg_path,
        episode_duration_s=episode_duration_s,
        seed=seed,
        use_gui=True,
        backend="traci",
        sumo_extra_args=["--delay", str(delay_ms)],
    )

    print(f"Loaded checkpoint {checkpoint_path} from training episode {episode}.")
    print("sumo-gui is running DQN greedy control. Close the GUI to stop early.")

    obs, info = env.reset(seed=seed)
    policy.reset()
    next_log_at = 0.0
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            action = policy.select_action(obs, info["sim_time_s"])
            obs, _reward, terminated, truncated, info = env.step(action)
            sim_time_s = float(info["sim_time_s"])
            if sim_time_s >= next_log_at:
                direction = Direction(action).name
                print(
                    f"t={sim_time_s:6.1f}s  action={direction:<5}  "
                    f"waiting={info['total_waiting_time_s']:8.1f}s  "
                    f"arrived={info['arrived_vehicles']}"
                )
                next_log_at = sim_time_s + log_every_s
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sumocfg", default="simulation/net/intersection.sumocfg")
    parser.add_argument("--episode-duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delay-ms", type=int, default=100)
    parser.add_argument("--log-every-s", type=float, default=30.0)
    args = parser.parse_args()

    run_gui(
        checkpoint_path=args.checkpoint,
        sumocfg_path=args.sumocfg,
        episode_duration_s=args.episode_duration,
        seed=args.seed,
        delay_ms=args.delay_ms,
        log_every_s=args.log_every_s,
    )


if __name__ == "__main__":
    main()
