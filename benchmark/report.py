"""Generate Stage 5 benchmark charts and a Markdown report from compare JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("docs/benchmarks")
SUMMARY_METRICS = (
    ("mean_waiting_time_s", "Mean waiting time (s)"),
    ("mean_queue_length", "Mean queue length"),
    ("arrived_vehicles", "Arrived vehicles"),
)


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_benchmark_artifacts(
    report_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    report = load_report(report_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_png = output_path / "benchmark_summary.png"
    improvement_png = output_path / "improvement_pct.png"
    markdown_path = output_path / "benchmark_report.md"

    _write_summary_chart(report, summary_png)
    _write_improvement_chart(report, improvement_png)
    markdown_path.write_text(
        _markdown_report(report, summary_png, improvement_png),
        encoding="utf-8",
    )

    return {
        "markdown": markdown_path,
        "summary_chart": summary_png,
        "improvement_chart": improvement_png,
    }


def _write_summary_chart(report: dict[str, Any], output_path: Path) -> None:
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    aggregated = report["aggregated"]
    labels = [label for _, label in SUMMARY_METRICS]
    fixed_values = [aggregated["fixed_time"][key] for key, _ in SUMMARY_METRICS]
    dqn_values = [aggregated["dqn"][key] for key, _ in SUMMARY_METRICS]
    x_positions = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x - width / 2 for x in x_positions], fixed_values, width, label="Fixed-time")
    ax.bar([x + width / 2 for x in x_positions], dqn_values, width, label="DQN")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Fixed-Time vs DQN Benchmark")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_improvement_chart(report: dict[str, Any], output_path: Path) -> None:
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    improvement = report["improvement_pct"]
    keys = ["mean_waiting_time_s", "mean_queue_length", "arrived_vehicles"]
    labels = ["Mean waiting", "Mean queue", "Throughput"]
    values = [improvement.get(key, 0.0) for key in keys]
    colors = ["#2e7d32" if value >= 0 else "#c62828" for value in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_ylabel("DQN improvement (%)")
    ax.set_title("DQN Improvement Over Fixed-Time")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _configure_matplotlib_cache() -> None:
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", ".tmp/matplotlib"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _markdown_report(report: dict[str, Any], summary_png: Path, improvement_png: Path) -> str:
    aggregated = report["aggregated"]
    improvement = report["improvement_pct"]
    seeds = ", ".join(str(seed) for seed in report.get("seeds", []))
    generated_at = report.get("generated_at", "unknown")

    rows = []
    for key, label in (
        ("mean_waiting_time_s", "Mean waiting time (s)"),
        ("final_waiting_time_s", "Final waiting time (s)"),
        ("mean_queue_length", "Mean queue length"),
        ("arrived_vehicles", "Arrived vehicles"),
    ):
        fixed_value = aggregated["fixed_time"][key]
        dqn_value = aggregated["dqn"][key]
        improvement_value = improvement.get(key, 0.0)
        rows.append(
            f"| {label} | {fixed_value:.2f} | {dqn_value:.2f} | {improvement_value:.1f}% |"
        )

    return "\n".join(
        [
            "# Benchmark Report",
            "",
            f"- Checkpoint: `{report.get('checkpoint', 'unknown')}`",
            f"- SUMO config: `{report.get('sumocfg_path', 'unknown')}`",
            f"- Episode duration: `{report.get('episode_duration_s', 'unknown')}` seconds",
            f"- Seeds: `{seeds}`",
            f"- Source generated at: `{generated_at}`",
            "",
            "## Headline",
            "",
            (
                "The trained DQN is compared against the fixed-time baseline using "
                "the same two-phase action space and the same yellow/all-red safety buffers."
            ),
            "",
            "## Metrics",
            "",
            "| Metric | Fixed-time | DQN | DQN improvement |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            "## Charts",
            "",
            f"![Benchmark summary]({summary_png.name})",
            "",
            f"![DQN improvement]({improvement_png.name})",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json", help="Path to benchmark.compare JSON output.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    paths = generate_benchmark_artifacts(args.report_json, args.output_dir)
    for artifact_name, path in paths.items():
        print(f"{artifact_name}: {path}")


if __name__ == "__main__":
    main()
