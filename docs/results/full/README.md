# Main study

The confirmatory evaluation: 6 conditions × 80 sequences × 8 attempts = 3,840 trials, plus 15
repeatability trials, collected from 2026-11-01 20:29 UTC to 2026-11-05 02:39 UTC with
`harness/experiment_config.full.yaml`.

| File | Contents |
|---|---|
| `analysis_report.json` | The analysis report |
| `trial_dataset.csv` | One row per trial |
| `uptime_events.json` | Service availability events |
| `repeatability.csv` | The 15 repeatability trials |
| `run_meta.json` | Run settings and provenance |
| `_run_manifest.json` | The run-defining configuration and its fingerprint, `58483d882d907717` |
| `schema.json` | The columns of `trial_dataset.csv` |

SHA-256:

- `trial_dataset.csv`: `bacc23c0e80a4152aeacc1662d8b21d4843a7ddd54ddae78f2829fc5127917a9`
- `uptime_events.json`: `3293579a903d5e2318e76fcfca2a94d34bf60ea32ebf5c46af4e3568ea69c08d`
- `repeatability.csv`: `52c3f3e3b870f17c960ad7f8722d6f0bfe94fbd02a93ecd89d68e44920f403ae`

The report was produced from the repository root with:

```bash
python cli.py analyze --dataset docs/results/full/trial_dataset.csv --reconfig-bench docs/results/benchmark/reconfig_cost.csv
```
