"""Manual smoke test for Stage 1: step the sim, print vehicle counts and the
traffic-light program at the controlled junction. Requires SUMO installed
and the network built (see simulation/net/build_net.py + generate_routes.py).

Usage:
    python -m simulation.traci_wrapper.smoke_test --steps 200 --gui
"""

from __future__ import annotations

import argparse

from common.logging import configure_logging, get_logger
from simulation.traci_wrapper import TraciSession

logger = get_logger(component="smoke_test")

TLS_ID = "C"  # matches the <node id="C" type="traffic_light"/> in intersection.nod.xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfg", default="simulation/net/intersection.sumocfg")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    configure_logging()

    max_vehicles_seen = 0
    with TraciSession(sumocfg_path=args.sumocfg, use_gui=args.gui) as sim:
        program = sim.traci.trafficlight.getRedYellowGreenState(TLS_ID)
        logger.info("tls.program", tls_id=TLS_ID, state=program)

        for step in range(args.steps):
            sim.step()
            vehicle_ids = sim.traci.vehicle.getIDList()
            max_vehicles_seen = max(max_vehicles_seen, len(vehicle_ids))
            if step % 50 == 0:
                logger.info("sim.tick", step=step, vehicles=len(vehicle_ids))

    logger.info("sim.done", max_vehicles_seen=max_vehicles_seen)
    assert max_vehicles_seen > 0, "No vehicles spawned - check intersection.rou.xml demand."
    print(f"OK: {args.steps} steps ran, peak concurrent vehicles = {max_vehicles_seen}")


if __name__ == "__main__":
    main()
