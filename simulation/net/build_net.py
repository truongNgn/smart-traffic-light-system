"""Compile intersection.nod.xml + intersection.edg.xml into intersection.net.xml
via SUMO's `netconvert`, then generate demand via generate_routes.py.

Requires SUMO installed and SUMO_HOME set (netconvert must be on PATH or under
$SUMO_HOME/bin). This mirrors exactly what Netedit does when you hit "save" —
run this instead of committing a hand-edited .net.xml so the network stays
regenerable from source.

Usage:
    python -m simulation.net.build_net
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

NET_DIR = Path(__file__).parent


def find_netconvert() -> str:
    exe = shutil.which("netconvert")
    if exe:
        return exe
    import os

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "netconvert"
        for suffix in ("", ".exe"):
            if (p := candidate.with_suffix(suffix)).exists():
                return str(p)
    raise RuntimeError(
        "netconvert not found. Install SUMO and either add its bin/ to PATH "
        "or set SUMO_HOME (e.g. C:\\Program Files (x86)\\Eclipse\\Sumo)."
    )


def build_network() -> Path:
    netconvert = find_netconvert()
    out_file = NET_DIR / "intersection.net.xml"
    cmd = [
        netconvert,
        "--node-files", str(NET_DIR / "intersection.nod.xml"),
        "--edge-files", str(NET_DIR / "intersection.edg.xml"),
        "--output-file", str(out_file),
        "--tls.guess", "true",
        "--tls.default-type", "static",
        "--no-turnarounds", "true",
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {out_file}")
    return out_file


def main() -> None:
    try:
        build_network()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
