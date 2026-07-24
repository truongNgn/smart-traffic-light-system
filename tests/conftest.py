"""Shared pytest fixtures/markers: skip SUMO-dependent tests cleanly when
SUMO isn't installed or the network hasn't been built, instead of failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simulation.traci_wrapper import SumoNotFoundError, find_sumo_binary

NET_DIR = Path(__file__).parent.parent / "simulation" / "net"
SUMOCFG = NET_DIR / "intersection.sumocfg"
# Routes-only config (no background <flow> demand) - for deterministic tests
# that inject their own vehicles and would otherwise be contaminated by
# randomly-generated traffic from the main intersection.rou.xml.
TEST_SUMOCFG = NET_DIR / "intersection_test.sumocfg"


def sumo_available() -> bool:
    try:
        find_sumo_binary(use_gui=False)
        return True
    except SumoNotFoundError:
        return False


def network_built() -> bool:
    return (NET_DIR / "intersection.net.xml").exists() and (NET_DIR / "intersection.rou.xml").exists()


requires_sumo = pytest.mark.skipif(not sumo_available(), reason="SUMO not installed / not on PATH")
requires_network = pytest.mark.skipif(
    not network_built(),
    reason="Network not built - run `python -m simulation.net.build_net` and "
    "`python -m simulation.net.generate_routes` first",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_routes_only_test_fixture() -> None:
    """intersection_test.rou.xml is a deterministic, generated file (no
    randomness - just <route> definitions, no <flow> demand) used by the
    golden grid-encoder tests to avoid contamination from background
    traffic. Generated files aren't committed (see .gitignore), so build it
    here on the fly rather than requiring a manual step before `pytest` works.
    """
    test_rou_path = NET_DIR / "intersection_test.rou.xml"
    if not network_built() or test_rou_path.exists():
        return
    from xml.etree.ElementTree import ElementTree, indent

    from simulation.net.generate_routes import build_rou_xml

    root = build_rou_xml(duration_s=0, seed=0, routes_only=True)
    tree = ElementTree(root)
    indent(tree, space="    ")
    tree.write(test_rou_path, encoding="UTF-8", xml_declaration=True)
