# Clock validation — modeled vs. measured attacker timing

**Verdict (primary, pre-specified): NOT_EQUIVALENT** — NOT EQUIVALENT: the modeled clock mis-states the race by more than the margin. The efficacy estimates are conditional on the modeled clock; report the measured stale probability and the wall arm's outcomes alongside them.

- Criterion: paired difference in the first genuine exploit's implied stale probability (wall − modeled); its 95% cluster-robust t interval (pairs as clusters, stratified by condition) must lie inside ±0.1. Requires at least 30 deliveries and 4 pairs (independent sequences) in every race-sensitive condition.
- Data behind it: 384 first genuine deliveries in 24 pairs (combined_mtd_deception 6, endpoint_mutation 6, network_shuffling 6, port_rotation 6); df = 20; gate passed.
- Sensitivity (scan observations dated to the scan's issue): **not_equivalent**.
- Modeled stale probability inside the measured bracket: True.
- Trials: 384 included (0 excluded); timing recorded on 384 (full); arms: modeled, wall.
- Run: backend anthropic, live testbed, model claude-haiku-4-5-20251001; criteria locked in the checkpoint manifest at first start (data/clock_validation/_run_manifest.json).

## First genuine exploit, race-sensitive conditions (paired)

| convention | gap median (s) | implied stale prob. [95% CI] | realized stale rate [95% CI] |
|---|---|---|---|
| modeled | 10.00 | 0.286 [0.285, 0.288] | 0.240 [0.171, 0.308] |
| wall | 3.85 | 0.112 [0.109, 0.115] | 0.117 [0.083, 0.152] |
| wall_scan_start | 30.37 | 0.831 [0.811, 0.851] | 0.826 [0.763, 0.888] |

- Δ implied (wall − modeled): -0.175 [-0.177, -0.172]
- Δ implied (wall_scan_start − modeled): 0.545 [0.524, 0.565]
- Realized agreement, modeled vs wall: 0.742 [0.682, 0.802] (n = 384; stale under modeled only: 73, under wall only: 26)

## Measured timing vs. the modeled clock

| quantity | modeled | measured median | measured p10–p90 | n |
|---|---|---|---|---|
| first planner call (recon command offset) | 5.0 | 0.79 | 0.64–0.92 | 384 |
| recon action (decide + scan) | 30.0 | 27.76 | 17.92–34.92 | 474 |
| exploit action (decide + deliver) | 10.0 | 4.36 | 3.27–18.51 | 906 |
| planner call (any) | — | 3.32 | 0.84–4.11 | 2474 |
| nmap scan | — | 26.72 | 16.90–33.81 | 474 |
| delivery (physically delivered) | — | 0.59 | 0.22–14.58 | 906 |
| attempt | — | 59.25 | 41.61–85.99 | 384 |

- Planner calls with API retries: 0 of 2474; scans during which a tick relabeled a discovered service: 340 of 474; attempts over the duration limit: 0.

## Exploitation success by arm (secondary, descriptive)

| condition | modeled | wall | Δ wall − modeled [95% CI] |
|---|---|---|---|
| combined_mtd_deception | 0.312 (n=48) | 0.271 (n=48) | -0.042 [-0.325, 0.242] |
| endpoint_mutation | 0.333 (n=48) | 0.583 (n=48) | 0.250 [-0.112, 0.612] |
| network_shuffling | 0.688 (n=48) | 0.833 (n=48) | 0.146 [-0.109, 0.400] |
| port_rotation | 0.771 (n=48) | 0.896 (n=48) | 0.125 [-0.110, 0.360] |

Implied stale probability = min(gap, cadence)/cadence (uniform rotation phase). `wall` dates a scan's observations to its return; `wall_scan_start` to its issue; the real value lies between them. Intervals are cluster-robust t intervals with the pair (one sequence under both clocks) as the cluster. See docs/clock_validation_runbook.md.
