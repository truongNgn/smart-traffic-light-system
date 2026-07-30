from __future__ import annotations

import json
from pathlib import Path

from benchmark.report import generate_benchmark_artifacts


def test_generate_benchmark_artifacts_writes_markdown_and_charts(tmp_path: Path) -> None:
    report_path = tmp_path / "comparison.json"
    report_path.write_text(
        json.dumps(
            {
                "checkpoint": "checkpoints/example.pt",
                "sumocfg_path": "simulation/net/intersection.sumocfg",
                "episode_duration_s": 60.0,
                "seeds": [1, 2],
                "generated_at": "2026-07-30T00:00:00+00:00",
                "aggregated": {
                    "fixed_time": {
                        "final_waiting_time_s": 10.0,
                        "mean_waiting_time_s": 20.0,
                        "mean_queue_length": 8.0,
                        "arrived_vehicles": 100.0,
                    },
                    "dqn": {
                        "final_waiting_time_s": 5.0,
                        "mean_waiting_time_s": 12.0,
                        "mean_queue_length": 6.0,
                        "arrived_vehicles": 110.0,
                    },
                },
                "improvement_pct": {
                    "final_waiting_time_s": 50.0,
                    "mean_waiting_time_s": 40.0,
                    "mean_queue_length": 25.0,
                    "arrived_vehicles": 10.0,
                },
            }
        ),
        encoding="utf-8",
    )

    paths = generate_benchmark_artifacts(report_path, tmp_path / "docs")

    assert paths["markdown"].exists()
    assert paths["summary_chart"].exists()
    assert paths["improvement_chart"].exists()
    assert "DQN improvement" in paths["markdown"].read_text(encoding="utf-8")
