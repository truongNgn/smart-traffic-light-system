"""Per-episode traffic metrics: the numbers Stage 5's benchmark report
actually compares (queue length reduction, waiting-time reduction,
throughput) - see docs referenced from IMPLEMENTATION_PLAN.md's Stage 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    """Aggregated over one full episode. `queue_length` here means the
    count of occupied cells in the 80-cell grid (simulation/state/
    grid_encoder.py) at each step - a proxy for vehicles queued, cheap to
    compute since GridEncoder already produces it every step."""

    total_steps: int = 0
    final_waiting_time_s: float = 0.0
    mean_waiting_time_s: float = 0.0
    max_waiting_time_s: float = 0.0
    mean_queue_length: float = 0.0
    max_queue_length: int = 0
    arrived_vehicles: int = 0
    episode_duration_s: float = 0.0

    _waiting_time_samples: list[float] = field(default_factory=list, repr=False)
    _queue_length_samples: list[int] = field(default_factory=list, repr=False)

    def record_step(self, waiting_time_s: float, queue_length: int, sim_time_s: float) -> None:
        self._waiting_time_samples.append(waiting_time_s)
        self._queue_length_samples.append(queue_length)
        self.total_steps += 1
        self.final_waiting_time_s = waiting_time_s
        self.episode_duration_s = sim_time_s

    def finalize(self, arrived_vehicles: int) -> "EpisodeMetrics":
        self.arrived_vehicles = arrived_vehicles
        if self._waiting_time_samples:
            self.mean_waiting_time_s = sum(self._waiting_time_samples) / len(
                self._waiting_time_samples
            )
            self.max_waiting_time_s = max(self._waiting_time_samples)
        if self._queue_length_samples:
            self.mean_queue_length = sum(self._queue_length_samples) / len(
                self._queue_length_samples
            )
            self.max_queue_length = max(self._queue_length_samples)
        return self

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "episode_duration_s": self.episode_duration_s,
            "final_waiting_time_s": self.final_waiting_time_s,
            "mean_waiting_time_s": self.mean_waiting_time_s,
            "max_waiting_time_s": self.max_waiting_time_s,
            "mean_queue_length": self.mean_queue_length,
            "max_queue_length": self.max_queue_length,
            "arrived_vehicles": self.arrived_vehicles,
        }
