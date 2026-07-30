# Benchmark Artifacts

Generated benchmark reports and charts live here.

Create or refresh them from a comparison JSON:

```bash
python -m benchmark.report benchmark/results/two_phase_eval_best_1200s_10seeds.json
```

The report generator writes:

- `benchmark_report.md`
- `benchmark_summary.png`
- `improvement_pct.png`
