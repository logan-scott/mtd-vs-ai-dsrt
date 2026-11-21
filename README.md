# Moving target defense against an LLM-planned attacker

Research instrument, testbed definitions, and data for the dissertation *Evaluating the Efficacy of
Moving Target Defense Strategies Against AI-Enabled Exploitation* (Logan Scott, University of the
Cumberlands).

The study measured whether five external moving target defense (MTD) conditions (network shuffling,
port rotation, endpoint mutation, deceptive presentation, and combined MTD-deception) reduce the
success of an attacker planned by a large language model, compared with a static baseline. The
planner (`claude-haiku-4-5-20251001`, through the Anthropic API) chose each reconnaissance and
exploitation step, and standard tools (nmap, curl, and Metasploit modules) carried the steps out
inside an isolated Proxmox testbed of five intentionally vulnerable services and three decoys.

## Repository layout

| Path | Contents |
|---|---|
| `cli.py` | Command-line entry point for data collection, analysis, and checks |
| `attacker/` | The LLM planner, the Anthropic client, the attacker's tools and system prompt, and an offline mock attacker |
| `harness/` | Experiment runner, live-testbed adapters, isolation preflight, provisioning, the reconfiguration benchmark, and the study configurations |
| `mtd/` | MTD techniques, the orchestrator, ground truth, and configuration checksums |
| `measurement/` | Trial metrics, timing, and the run sheet that defines the trial dataset |
| `analysis/` | Statistical analysis for the five research questions, the power analysis, and the sensitivity checks |
| `testbed/` | Service roster, vulnerability profile, and the deception responder |
| `infra/` | Terraform and Ansible definitions of the isolated testbed (see `infra/README.md`) |
| `scripts/` | Deployment, image pinning, benchmark staging, and the supplementary checks |
| `docs/results/` | The study's data |

## Data

Each folder has a README that describes the run and its files.

| Folder | Contents | Collected (UTC) |
|---|---|---|
| `docs/results/full/` | Main study: 6 conditions × 80 sequences × 8 attempts = 3,840 trials, plus 15 repeatability trials | 2026-11-01 20:29 to 2026-11-05 02:39 |
| `docs/results/full_fastscan/` | Fast-scan check: the static baseline and the four conditions that move genuine services, 20 sequences each (800 trials), with the scan aimed at the target host | 2026-11-05 03:22 to 10:37 |
| `docs/results/benchmark/` | Reconfiguration-cost benchmark: 6 techniques × 20 rounds on the target host | 2026-11-05 |
| `docs/results/pilot/` | First pilot: 32 trials in two conditions, used for planning values only | 2026-07-21 |
| `docs/results/clock_validation/` | Timing pilot: 384 trials comparing the earlier modeled attacker clock with the real clock | 2026-10-26 to 2026-10-27 |

Each `trial_dataset.csv` has one row per trial. The `schema.json` beside it lists the columns, and
Appendix F of the dissertation defines every variable.

## Reproducing the analysis

The reports were produced with Python 3.13.0 and the library versions pinned in
`requirements-analysis.lock`. From the repository root:

```bash
python3.13 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.lock
.venv-analysis/bin/python cli.py analyze --dataset docs/results/full/trial_dataset.csv --reconfig-bench docs/results/benchmark/reconfig_cost.csv --out data/full_analysis_report.json
.venv-analysis/bin/python cli.py analyze --dataset docs/results/full_fastscan/trial_dataset.csv --reconfig-bench docs/results/benchmark/reconfig_cost.csv --out data/full_fastscan_analysis_report.json
.venv-analysis/bin/python cli.py fastscan-compare --main docs/results/full/trial_dataset.csv --fast docs/results/full_fastscan/trial_dataset.csv --out data/fastscan_comparison.json
```

A few values in the dissertation come from the main study's data but are not in the analysis
report: the treatment-delivery checks, the replies that carried no action, the exploratory
observations reported with the third research question, and the cross-validation folds of the
fourth research question's regularized selection. To print them:

```bash
.venv-analysis/bin/python scripts/supplementary_checks.py
```

## Collecting data

New data need the testbed and an Anthropic API key. Build the testbed as described in
`infra/README.md`, copy `.secrets.example` to `.secrets` and add the key, and then, on the control
host:

```bash
python cli.py preflight --config harness/experiment_config.full.yaml
python cli.py run --live --backend anthropic --config harness/experiment_config.full.yaml --outdir data/full
python cli.py run --live --backend anthropic --config harness/experiment_config.full_fastscan.yaml --outdir data/full_fastscan
```

The reconfiguration benchmark runs on the target host; `scripts/stage_reconfig_bench.sh` copies the
code there and prints the commands.

## Ethics and containment

The attacks ran only inside an isolated lab owned by the researcher. The testbed and attacker nodes
share an internal bridge with no uplink and no gateway; only the control host, which runs the
planner and has IP forwarding turned off, reaches the model API. A fail-closed preflight
(`harness/preflight.py`) checks this isolation before every live run. The targets are public,
intentionally vulnerable images, and the exploits are standard public tools; the repository contains
no novel exploit or malware code. Run this code only against systems you own and have isolated.

## License

MIT; see `LICENSE`.
