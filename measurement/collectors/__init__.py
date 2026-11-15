"""Collectors: deterministic parsers from raw/fixture logs into typed models.

In the offline path these read JSONL/JSON fixtures; the same parsers will
back the live collectors once the testbed exists. Parsing is deterministic and
preserves source ordering.
"""

from measurement.collectors.parsers import load_trial_inputs

__all__ = ["load_trial_inputs"]
