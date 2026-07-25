"""Drives one full episode under a given Policy, collecting EpisodeMetrics
along the way. Shared by both sides of the comparison in compare.py -
identical env, identical metric collection, only the policy differs.
"""

from __future__ import annotations

import numpy as np

from benchmark.metrics import EpisodeMetrics
from benchmark.policies import Policy
from rl.env.traffic_env import SumoTrafficEnv


def run_episode(env: SumoTrafficEnv, policy: Policy, seed: int | None = None) -> EpisodeMetrics:
    obs, info = env.reset(seed=seed)
    policy.reset()

    metrics = EpisodeMetrics()
    metrics.record_step(
        waiting_time_s=info["total_waiting_time_s"],
        queue_length=int(np.sum(obs)),
        sim_time_s=info["sim_time_s"],
    )
    arrived_vehicles = info["arrived_vehicles"]

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.select_action(obs, info["sim_time_s"])
        obs, _reward, terminated, truncated, info = env.step(action)
        metrics.record_step(
            waiting_time_s=info["total_waiting_time_s"],
            queue_length=int(np.sum(obs)),
            sim_time_s=info["sim_time_s"],
        )
        arrived_vehicles = info["arrived_vehicles"]

    return metrics.finalize(arrived_vehicles=arrived_vehicles)
