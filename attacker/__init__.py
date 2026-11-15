"""AI-enabled offensive workflow — pluggable, config-driven attacker adapters.

The ``Attacker`` interface (``attacker.base``) is backend-agnostic: an attempt
takes an ``Environment`` it can probe and a retained ``AttackerContext``, and
returns the recon/exploit observations the measurement instrument consumes. The
offline ``MockAttacker`` (``attacker.mock``) is a deterministic stand-in for
producing synthetic datasets without live infrastructure or LLM calls; the real
LLM-backed adapter (``attacker.planner.LLMAttacker``) implements the same interface.
"""

from attacker.base import (
    ActionTraceEntry,
    AttackerContext,
    AttemptObservation,
    AttemptRequest,
    Environment,
)
from attacker.errors import AttackerApiError

__all__ = [
    "ActionTraceEntry",
    "AttackerContext",
    "AttemptObservation",
    "AttemptRequest",
    "Environment",
    "AttackerApiError",
]
