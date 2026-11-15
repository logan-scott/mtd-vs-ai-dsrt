"""Experiment harness: trial_runner + experiment_runner.

Drives the full pipeline — MTD orchestrator → attacker → measurement — to produce
the trial-level dataset. Offline it uses the mock attacker and synthetic telemetry
so a complete, populated dataset can be generated without infrastructure; the same
runners will drive the live testbed once provisioned.
"""
