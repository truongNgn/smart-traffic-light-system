from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import EpisodeMetrics
from benchmark.policies import DQNPolicy, FixedTimePolicy
from benchmark.run_episode import run_episode
from rl.agent.dqn_agent import DQNAgent
from rl.env.traffic_env import SumoTrafficEnv
from rl.train.checkpoint import load_checkpoint


GUARD_CONFIG = dict(
    max_red_time_s=None,
    soft_red_time_s=70.0,
    hard_red_time_s=120.0,
    starving_queue_threshold=4,
    starving_wait_time_s=160.0,
)


def flow_origin(flow: ET.Element) -> str:
    route = flow.get("route", "")
    if route.startswith("route_"):
        return route.split("_")[1]
    parts = flow.get("id", "").split("_")
    return parts[-2] if len(parts) >= 2 else ""


def write_sumocfg(path: Path, route_file: str) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="{route_file}"/>
        <additional-files value="vtypes.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <duration-log.disable value="true"/>
    </report>
</configuration>
""",
        encoding="utf-8",
    )


def scaled_rate(flow: ET.Element, *, all_scale: float = 1.0, ew_scale: float = 1.0, ns_scale: float = 1.0) -> float:
    origin = flow_origin(flow)
    axis_scale = ew_scale if origin in {"E", "W"} else ns_scale
    return float(flow.get("vehsPerHour")) * all_scale * axis_scale


def make_scaled_scenario(
    name: str,
    *,
    duration_s: int = 1800,
    all_scale: float = 1.0,
    ew_scale: float = 1.0,
    ns_scale: float = 1.0,
) -> str:
    src = Path("simulation/net/intersection.rou.xml")
    route_path = Path(f"simulation/net/{name}.rou.xml")
    sumocfg_path = Path(f"simulation/net/{name}.sumocfg")
    tree = ET.parse(src)
    root = tree.getroot()

    for flow in root.findall("flow"):
        flow.set("end", str(duration_s))
        flow.set(
            "vehsPerHour",
            f"{scaled_rate(flow, all_scale=all_scale, ew_scale=ew_scale, ns_scale=ns_scale):.2f}",
        )

    ET.indent(tree, space="    ")
    tree.write(route_path, encoding="UTF-8", xml_declaration=True)
    write_sumocfg(sumocfg_path, route_path.name)
    return str(sumocfg_path)


def make_time_block_scenario(name: str, blocks: list[tuple[str, int, int, dict]]) -> str:
    src = Path("simulation/net/intersection.rou.xml")
    route_path = Path(f"simulation/net/{name}.rou.xml")
    sumocfg_path = Path(f"simulation/net/{name}.sumocfg")
    base_tree = ET.parse(src)
    base_root = base_tree.getroot()

    root = ET.Element(base_root.tag, base_root.attrib)
    for route in base_root.findall("route"):
        root.append(deepcopy(route))

    for block_name, begin, end, multipliers in blocks:
        for flow in base_root.findall("flow"):
            block_flow = deepcopy(flow)
            block_flow.set("id", f"{flow.get('id')}_{block_name}")
            block_flow.set("begin", str(begin))
            block_flow.set("end", str(end))
            block_flow.set("vehsPerHour", f"{scaled_rate(flow, **multipliers):.2f}")
            root.append(block_flow)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(route_path, encoding="UTF-8", xml_declaration=True)
    write_sumocfg(sumocfg_path, route_path.name)
    return str(sumocfg_path)


def build_scenarios() -> tuple[dict[str, str], dict[str, int]]:
    scenarios = {
        "normal": "simulation/net/intersection.sumocfg",
        "heavy_2x": make_scaled_scenario("intersection_heavy_2x", all_scale=2.0),
        "imbalanced_ew3x": make_scaled_scenario("intersection_imbalanced_ew3x", ew_scale=3.0),
        "imbalanced_ns3x": make_scaled_scenario("intersection_imbalanced_ns3x", ns_scale=3.0),
    }
    scenarios["mixed_balanced_peak"] = make_time_block_scenario(
        "intersection_mixed_balanced_peak",
        [
            ("normal", 0, 600, dict(all_scale=1.0)),
            ("heavy2x", 600, 1200, dict(all_scale=2.0)),
            ("ew3x", 1200, 1800, dict(ew_scale=3.0)),
            ("ns3x", 1800, 2400, dict(ns_scale=3.0)),
        ],
    )
    durations = {
        "normal": 1200,
        "heavy_2x": 1800,
        "imbalanced_ew3x": 1800,
        "imbalanced_ns3x": 1800,
        "mixed_balanced_peak": 2400,
    }
    return scenarios, durations


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(metrics: list[EpisodeMetrics]) -> dict[str, float]:
    dicts = [metric.to_dict() for metric in metrics]
    return {key: mean([item[key] for item in dicts]) for key in dicts[0]}


def improvement_pct(fixed: dict[str, float], dqn: dict[str, float]) -> dict[str, float]:
    return {
        "mean_wait": (fixed["mean_waiting_time_s"] - dqn["mean_waiting_time_s"]) / fixed["mean_waiting_time_s"] * 100,
        "final_wait": (fixed["final_waiting_time_s"] - dqn["final_waiting_time_s"]) / fixed["final_waiting_time_s"] * 100 if fixed["final_waiting_time_s"] else 0.0,
        "queue": (fixed["mean_queue_length"] - dqn["mean_queue_length"]) / fixed["mean_queue_length"] * 100,
        "arrived": (dqn["arrived_vehicles"] - fixed["arrived_vehicles"]) / fixed["arrived_vehicles"] * 100,
    }


def evaluate_policy(scenario: str, duration_s: int, policy, seeds: list[int], backend: str) -> list[EpisodeMetrics]:
    env = SumoTrafficEnv(
        sumocfg_path=scenario,
        episode_duration_s=duration_s,
        green_duration_s=10.0,
        backend=backend,
        **GUARD_CONFIG,
    )
    try:
        return [run_episode(env, policy, seed=seed) for seed in seeds]
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dqn_eval_best.pt")
    parser.add_argument("--backend", choices=["traci", "libsumo"], default="traci")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scenarios, durations = build_scenarios()
    agent = DQNAgent()
    trained_episode = load_checkpoint(args.checkpoint, agent)
    dqn_policy = DQNPolicy(agent, epsilon=0.0)
    fixed_policy = FixedTimePolicy(green_duration_s=20.0)

    report = {
        "checkpoint": args.checkpoint,
        "trained_episode": trained_episode,
        "seeds": args.seeds,
        "guard_config": GUARD_CONFIG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": {},
    }

    print(f"checkpoint={args.checkpoint} episode={trained_episode} seeds={args.seeds}")
    print(f"{'scenario':<24}{'mean_wait':>11}{'final_wait':>12}{'queue':>10}{'arrived':>10}")
    for name, sumocfg in scenarios.items():
        fixed_runs = evaluate_policy(sumocfg, durations[name], fixed_policy, args.seeds, args.backend)
        dqn_runs = evaluate_policy(sumocfg, durations[name], dqn_policy, args.seeds, args.backend)
        fixed_agg = aggregate(fixed_runs)
        dqn_agg = aggregate(dqn_runs)
        imp = improvement_pct(fixed_agg, dqn_agg)
        report["scenarios"][name] = {
            "duration_s": durations[name],
            "fixed_time": fixed_agg,
            "dqn": dqn_agg,
            "improvement_pct": imp,
            "per_seed": {
                "fixed_time": [metric.to_dict() for metric in fixed_runs],
                "dqn": [metric.to_dict() for metric in dqn_runs],
            },
        }
        print(
            f"{name:<24}{imp['mean_wait']:>10.1f}%{imp['final_wait']:>11.1f}%"
            f"{imp['queue']:>9.1f}%{imp['arrived']:>9.1f}%"
        )

    output = Path(args.output) if args.output else Path("benchmark/results") / (
        f"stress_{Path(args.checkpoint).stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = Path("benchmark/results/latest_stress_report.json")
    shutil.copy2(output, latest)
    print(f"saved={output}")
    print(f"latest={latest}")


if __name__ == "__main__":
    main()
