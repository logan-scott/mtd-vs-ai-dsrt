"""Typed attacker errors."""

from __future__ import annotations


class AttackerApiError(RuntimeError):
    """The attacker's model API failed unrecoverably (after SDK + app-level retries).

    Raised by the LLM backend when a transient API condition (overload / rate-limit /
    connection / timeout) outlasts both the SDK's built-in retries and the thin
    application-level backoff wrapper. It is a *typed* error so the experiment runner
    can classify a transient outage distinctly from a genuine bug and apply the
    configured fault policy (exclude the sequence and continue, or halt for resume).
    """
