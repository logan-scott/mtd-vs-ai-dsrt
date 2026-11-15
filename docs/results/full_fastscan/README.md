# Fast-scan check

The planned sensitivity check of the main study, run on the same code and settings except for the
scan. Each reconnaissance scan was pointed at the target host from the start
(`live.scan_scope: target_host`) instead of sweeping the testbed subnet, so the attacker began every
attempt knowing where the services were. That is information the main study's attacker had to find
by sweeping the subnet, so the check is deliberately favorable to the attacker.

The static baseline and the four conditions that move genuine services × 20 sequences × 8 attempts
= 800 trials, plus 15 repeatability trials, collected from 2026-11-05 03:22 to 10:37 UTC with
`harness/experiment_config.full_fastscan.yaml`.

| File | Contents |
|---|---|
| `fastscan_comparison.json` | The planned comparison with the main study: for each shared condition, the proportion of trials compromised and the restricted mean time to compromise over the first 300 s under each scan, their difference, and the change in the moving conditions' benefit over the static baseline, with 95% cluster-bootstrap intervals |
| `analysis_report.json` | The fast-scan run's own analysis report |
| `trial_dataset.csv`, `uptime_events.json`, `repeatability.csv`, `run_meta.json`, `schema.json` | As in `../full/` |
| `_run_manifest.json` | The run-defining configuration and its fingerprint, `a87dd8519c04f706` |

SHA-256:

- `trial_dataset.csv`: `1368751731d5d3278df93afaaea3c7838343aeba09bc278afeacab3aeacaa59b`
- `uptime_events.json`: `adbe0f3a0102573896e68348c3fcfe7cf3269341c6a1de01dba7723b4dc3e997`
- `repeatability.csv`: `3d4080ba2db6732def9f5e0c32ed9ef07c1687084d97e8cf3961ef573c0e7db6`

The reports were produced from the repository root with:

```bash
python cli.py analyze --dataset docs/results/full_fastscan/trial_dataset.csv --reconfig-bench docs/results/benchmark/reconfig_cost.csv
python cli.py fastscan-compare --main docs/results/full/trial_dataset.csv --fast docs/results/full_fastscan/trial_dataset.csv
```
