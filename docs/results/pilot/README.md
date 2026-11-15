# First pilot

Run on 2026-07-21 under the institutional review board's pilot approval: the static baseline and
port rotation, 2 sequences × 8 attempts each, 32 trials. It supplied planning values only; no pilot
data enter the confirmatory analyses. Its run-defining settings are recorded on every row of
`trial_dataset.csv`.

| File | Contents |
|---|---|
| `analysis_report.json` | The report written at the end of the pilot |
| `trial_dataset.csv` | One row per trial |
| `uptime_events.json`, `repeatability.csv` | Service availability events and the 15 repeatability trials |
| `run_meta.json`, `schema.json` | Run summary and the column schema |

The power analysis (`analysis/power.py`) draws its planning value from these data; see the `power`
section of `../full/analysis_report.json`.
