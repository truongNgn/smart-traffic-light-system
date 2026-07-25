# Benchmarking: Fixed-Time vs DQN

Stage 4's third Engineer-B deliverable (after the training pipeline and
experiment tracking): compare the trained agent against a classic
fixed-time baseline on the same network and demand.

## Running it

Once you have a checkpoint (`checkpoints/dqn_best.pt` from a real Kaggle
training run, or `dqn_final.pt`):

```bash
python -m benchmark.compare --checkpoint checkpoints/dqn_best.pt
```

Defaults: `simulation/net/intersection.sumocfg`, 3600s episodes, seeds
`[1, 2, 3]`, 20s fixed-time green phases. Override any of them:

```bash
python -m benchmark.compare \
    --checkpoint checkpoints/dqn_best.pt \
    --episode-duration 3600 \
    --seeds 1 2 3 4 5 \
    --fixed-green-duration 20
```

This prints a comparison table and writes the full per-seed + aggregated
results to `benchmark/results/comparison_<timestamp>.json` (gitignored -
regenerate any time, and Stage 5's charts will read from here).

## What's being compared

Both policies drive the *same* `SumoTrafficEnv` from Stage 3 - the
fixed-time controller only decides which direction gets the next green
phase; `SumoTrafficEnv` is still the only place the yellow+all-red safety
buffer gets inserted (`benchmark/policies.py`'s module docstring explains
why this matters: it keeps the comparison apples-to-apples instead of
giving one side a timing advantage).

- **`FixedTimePolicy`** (`benchmark/policies.py`): cycles East → North →
  West → South, each held green for `--fixed-green-duration` seconds -
  the paper's baseline for comparison.
- **`DQNPolicy`**: the trained agent, greedy (`epsilon=0`) - no
  exploration noise in a benchmark run.

## Metrics collected (`benchmark/metrics.py`)

| Metric | Meaning |
|---|---|
| `mean_waiting_time_s` / `final_waiting_time_s` | Sum of per-vehicle waiting time across the network, sampled every step / at episode end |
| `mean_queue_length` / `max_queue_length` | Count of occupied cells in the 80-cell grid (proxy for queued vehicles) |
| `arrived_vehicles` | Throughput - vehicles that completed their route during the episode |

`improvement_pct` in the JSON report is signed so positive always means
"DQN is better" (lower waiting/queue, higher throughput).

## Sanity-checking without a trained model

`tests/test_benchmark.py` runs the whole pipeline (including `compare()`
end-to-end) against a freshly-initialized, *untrained* `DQNAgent` - it
will not beat the baseline, and that's expected; those tests only prove
the wiring is correct. Don't read anything into an untrained checkpoint's
numbers.
