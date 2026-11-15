# Timing pilot (clock validation)

This pilot tested whether the instrument's earlier modeled clock, which charged each action a fixed
duration, matched the attacker's real timing. `cli.py validate-clock` ran every sequence twice with
the same seed, once under each clock, in the four conditions that move genuine services:
4 conditions × 6 sequences × 8 attempts × 2 clocks = 384 trials, collected on 2026-10-26 and
2026-10-27. Its result led the main study to time every action on the real clock, and its
real-clock arm (`arm_wall/trial_dataset.csv`) gives the power analysis a second clustering estimate.

| File | Contents |
|---|---|
| `clock_validation_report.json`, `clock_validation_summary.md` | The validation report and its summary |
| `trial_dataset.csv`, `uptime_events.json` | Both arms combined |
| `arm_modeled/`, `arm_wall/` | Each arm's rows, uptime events, run metadata, and schema |
| `run_meta.json`, `_run_manifest.json`, `schema.json`, `code_version.txt` | Settings, the pre-specified criteria, the column schema, and the code version |
