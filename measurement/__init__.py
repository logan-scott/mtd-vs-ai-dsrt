"""Measurement instrument — deterministic collectors + metric modules.

This package computes every dependent and control variable defined in
``docs/metrics_specification.md`` for one trial, and assembles the trial-level
dataset row. All metric computation is deterministic: identical inputs produce
identical outputs (a reliability requirement, not an implementation detail).
"""

__all__ = [
    "models",
    "clock",
    "correlation",
    "compute_trial",
    "run_sheet",
]
