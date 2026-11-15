# Reconfiguration-cost benchmark

`cli.py reconfig-bench`, run on the target host on 2026-11-05 from the same code as the main study:
6 techniques × 20 rounds, each reconfiguration timed after every published service was up.
`reconfig_cost.csv` has one row per round, `summary.json` the per-technique summaries, and
`metadata.json` the settings, host, and image digests. The analysis of research questions four and
five reads `reconfig_cost.csv`.

SHA-256 of `reconfig_cost.csv`: `90db6ad46cd8e8531c543417599e3612c10c8cd2b7f3690292cdfbd71b594061`.

To run the benchmark again, stage the code on the target host with `scripts/stage_reconfig_bench.sh`
and follow the commands it prints.
