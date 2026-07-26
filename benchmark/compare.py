"""Fixed-time baseline vs trained DQN comparison - Stage 4's third
Engineer-B deliverable (training pipeline + experiment tracking + this).

Usage:
    python -m benchmark.compare --checkpoint checkpoints/dqn_best.pt

Runs both policies across the same set of seeds on the same network/demand,
prints a side-by-side comparison with percent improvement, and writes the
full results (per-seed and aggregated) to a JSON file under
benchmark/results/ for Stage 5's charts/report to consume later.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common.logging import configure_logging, get_logger
from benchmark.metrics import EpisodeMetrics
from benchmark.policies import DQNPolicy, FixedTimePolicy
from benchmark.run_episode import run_episode
from rl.agent.dqn_agent import DQNAgent
from rl.env.traffic_env import SumoTrafficEnv
from rl.train.checkpoint import load_checkpoint

logger = get_logger(component="benchmark")

RESULTS_DIR = Path(__file__).parent / "results"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(per_seed: list[EpisodeMetrics]) -> dict:
    dicts = [m.to_dict() for m in per_seed]
    keys = dicts[0].keys() if dicts else []
    return {key: _mean([d[key] for d in dicts]) for key in keys}


def compare(
    *,
    checkpoint_path: str,
    sumocfg_path: str = "simulation/net/intersection.sumocfg",
    episode_duration_s: float = 3600.0,
    seeds: list[int] = (1, 2, 3),
    fixed_green_duration_s: float = 20.0,
    backend: str = "libsumo",
    use_gui: bool = False,
) -> dict:
    configure_logging()

    if use_gui:
        # sumo-gui only exists under the traci backend - libsumo is
        # in-process and headless by construction (see
        # simulation/traci_wrapper/session.py).
        backend = "traci"
    else:
        try:
            import libsumo  # noqa: F401
        except ImportError:
            if backend == "libsumo":
                logger.warning("benchmark.libsumo_not_installed_falling_back_to_traci")
                backend = "traci"

    agent = DQNAgent()
    episode = load_checkpoint(checkpoint_path, agent)
    logger.info("benchmark.checkpoint_loaded", checkpoint=checkpoint_path, trained_episode=episode)

    env = SumoTrafficEnv(
        sumocfg_path=sumocfg_path,
        episode_duration_s=episode_duration_s,
        backend=backend,
        use_gui=use_gui,
    )

    fixed_time_policy = FixedTimePolicy(green_duration_s=fixed_green_duration_s)
    dqn_policy = DQNPolicy(agent, epsilon=0.0)

    results: dict[str, list[EpisodeMetrics]] = {"fixed_time": [], "dqn": []}
    try:
        for seed in seeds:
            logger.info("benchmark.run_start", policy="fixed_time", seed=seed)
            results["fixed_time"].append(run_episode(env, fixed_time_policy, seed=seed))

            logger.info("benchmark.run_start", policy="dqn", seed=seed)
            results["dqn"].append(run_episode(env, dqn_policy, seed=seed))
    finally:
        env.close()

    aggregated = {name: _aggregate(runs) for name, runs in results.items()}
    improvement = _compute_improvement(aggregated["fixed_time"], aggregated["dqn"])

    report = {
        "checkpoint": checkpoint_path,
        "trained_episode": episode,
        "sumocfg_path": sumocfg_path,
        "episode_duration_s": episode_duration_s,
        "seeds": list(seeds),
        "fixed_green_duration_s": fixed_green_duration_s,
        "aggregated": aggregated,
        "improvement_pct": improvement,
        "per_seed": {
            name: [m.to_dict() for m in runs] for name, runs in results.items()
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _print_summary(aggregated, improvement)
    return report


def _compute_improvement(fixed_time: dict, dqn: dict) -> dict:
    """Positive = DQN better. Waiting time and queue length: lower is
    better, so improvement = (fixed - dqn) / fixed. Throughput: higher is
    better, so improvement = (dqn - fixed) / fixed."""

    def pct_lower_is_better(fixed_val: float, dqn_val: float) -> float:
        if fixed_val == 0:
            return 0.0
        return (fixed_val - dqn_val) / fixed_val * 100.0

    def pct_higher_is_better(fixed_val: float, dqn_val: float) -> float:
        if fixed_val == 0:
            return 0.0
        return (dqn_val - fixed_val) / fixed_val * 100.0

    return {
        "final_waiting_time_s": pct_lower_is_better(
            fixed_time["final_waiting_time_s"], dqn["final_waiting_time_s"]
        ),
        "mean_waiting_time_s": pct_lower_is_better(
            fixed_time["mean_waiting_time_s"], dqn["mean_waiting_time_s"]
        ),
        "mean_queue_length": pct_lower_is_better(
            fixed_time["mean_queue_length"], dqn["mean_queue_length"]
        ),
        "arrived_vehicles": pct_higher_is_better(
            fixed_time["arrived_vehicles"], dqn["arrived_vehicles"]
        ),
    }


def _print_summary(aggregated: dict, improvement: dict) -> None:
    print("\n=== Fixed-Time vs DQN (averaged across seeds) ===")
    print(f"{'metric':<24}{'fixed-time':>14}{'dqn':>14}{'improvement':>14}")
    rows = [
        ("mean waiting time (s)", "mean_waiting_time_s"),
        ("final waiting time (s)", "final_waiting_time_s"),
        ("mean queue length", "mean_queue_length"),
        ("arrived vehicles", "arrived_vehicles"),
    ]
    for label, key in rows:
        fixed_val = aggregated["fixed_time"][key]
        dqn_val = aggregated["dqn"][key]
        pct = improvement.get(key, 0.0)
        print(f"{label:<24}{fixed_val:>14.2f}{dqn_val:>14.2f}{pct:>13.1f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--sumocfg", dest="sumocfg_path", default="simulation/net/intersection.sumocfg")
    parser.add_argument("--episode-duration", dest="episode_duration_s", type=float, default=3600.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--fixed-green-duration", type=float, default=20.0)
    parser.add_argument("--backend", choices=["libsumo", "traci"], default="libsumo")
    parser.add_argument(
        "--gui", dest="use_gui", action="store_true", help="Watch it run in sumo-gui."
    )
    parser.add_argument("--output", default=None, help="Output JSON path (default: auto-named under benchmark/results/).")
    args = parser.parse_args()

    report = compare(
        checkpoint_path=args.checkpoint,
        sumocfg_path=args.sumocfg_path,
        episode_duration_s=args.episode_duration_s,
        seeds=args.seeds,
        fixed_green_duration_s=args.fixed_green_duration,
        use_gui=args.use_gui,
        backend=args.backend,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"comparison_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("benchmark.report_saved", path=str(output_path))
    print(f"Full report saved to {output_path}")


if __name__ == "__main__":
    main()
