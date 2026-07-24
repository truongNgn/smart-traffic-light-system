"""Generate intersection.rou.xml: demand for all 4 vehicle classes across all
12 turning movements (4 entrances x 3 possible exits) of the 4-way intersection.

Usage:
    python -m simulation.net.generate_routes --seed 42 --duration 3600

Kept as a script rather than a hand-written .rou.xml so demand volumes and
class mix are easy to tune without hand-editing 48 <flow> elements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

NET_DIR = Path(__file__).parent

# Entrance edge -> the 3 edges it can exit through (straight + two turns).
# Edge ids match intersection.edg.xml.
ENTRANCES: dict[str, str] = {"N2C": "N", "S2C": "S", "E2C": "E", "W2C": "W"}
EXITS: dict[str, str] = {"N": "C2N", "S": "C2S", "E": "C2E", "W": "C2W"}

# Per-class demand as vehicles/hour, on the busiest routes; scaled down for turns.
# Ratios roughly reflect the paper's mixed-traffic setting (car-dominant, light bus/truck).
CLASS_VOLUME_VPH: dict[str, float] = {
    "car": 360.0,
    "motorcycle": 240.0,
    "bus": 20.0,
    "truck": 30.0,
}

# Fraction of an entrance's total demand that goes straight vs turns (must sum to 1.0
# across the 3 reachable exits). Index 0 = straight, others = the two turns.
TURN_SPLIT = {"straight": 0.5, "left": 0.25, "right": 0.25}

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
# Right-of-90-degrees turn from each origin (used to label left vs right, cosmetic only).
RIGHT_TURN_FROM = {"N": "E", "E": "S", "S": "W", "W": "N"}


@dataclass(frozen=True)
class Route:
    id: str
    edges: str


def build_routes() -> list[Route]:
    routes: list[Route] = []
    for entrance_edge, origin in ENTRANCES.items():
        for dest in ("N", "S", "E", "W"):
            if dest == origin:
                continue
            exit_edge = EXITS[dest]
            routes.append(Route(id=f"route_{origin}_to_{dest}", edges=f"{entrance_edge} {exit_edge}"))
    return routes


def turn_kind(origin: str, dest: str) -> str:
    if dest == OPPOSITE[origin]:
        return "straight"
    if dest == RIGHT_TURN_FROM[origin]:
        return "right"
    return "left"


def build_rou_xml(duration_s: int, seed: int, routes_only: bool = False) -> Element:
    root = Element("routes")
    root.set(
        "xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"
    )

    routes = build_routes()
    for route in routes:
        SubElement(root, "route", id=route.id, edges=route.edges)

    if routes_only:
        return root

    flow_index = 0
    for entrance_edge, origin in ENTRANCES.items():
        for dest in ("N", "S", "E", "W"):
            if dest == origin:
                continue
            kind = turn_kind(origin, dest)
            route_id = f"route_{origin}_to_{dest}"
            split = TURN_SPLIT[kind]
            for vclass, base_vph in CLASS_VOLUME_VPH.items():
                vph = base_vph * split
                if vph <= 0:
                    continue
                flow_index += 1
                SubElement(
                    root,
                    "flow",
                    id=f"flow_{flow_index:03d}_{vclass}_{origin}_{dest}",
                    type=vclass,
                    route=route_id,
                    begin="0",
                    end=str(duration_s),
                    vehsPerHour=f"{vph:.2f}",
                    departLane="best",
                    departSpeed="max",
                )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=3600, help="Episode length in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=NET_DIR / "intersection.rou.xml", help="Output .rou.xml path."
    )
    parser.add_argument(
        "--routes-only",
        action="store_true",
        help="Emit <route> definitions only, no <flow> demand - used for deterministic tests "
        "that inject their own vehicles via traci.vehicle.add/moveTo.",
    )
    args = parser.parse_args()

    root = build_rou_xml(duration_s=args.duration, seed=args.seed, routes_only=args.routes_only)
    tree = ElementTree(root)
    indent(tree, space="    ")
    tree.write(args.out, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
