"""Script to export Pydantic schemas as JSON Schema definitions."""

import json
from pathlib import Path

from common.schemas.control import PhaseAction, PhaseState, ReasoningLog
from common.schemas.state import IntersectionState
from common.schemas.telemetry import Telemetry
from common.schemas.vision import FeedFrame, VehicleCountEvent

SCHEMAS = {
    "PhaseAction": PhaseAction,
    "PhaseState": PhaseState,
    "ReasoningLog": ReasoningLog,
    "IntersectionState": IntersectionState,
    "Telemetry": Telemetry,
    "FeedFrame": FeedFrame,
    "VehicleCountEvent": VehicleCountEvent,
}

def main() -> None:
    out_dir = Path("docs/schemas")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in SCHEMAS.items():
        schema_json = model.model_json_schema()
        file_path = out_dir / f"{name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(schema_json, f, indent=2)
            
        print(f"Exported {name} -> {file_path}")

if __name__ == "__main__":
    main()
