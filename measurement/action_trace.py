"""Compact, lossless codec for the planner's per-trial action sequence.

PURE INSTRUMENTATION. The trace records *which action the attacker's planner chose*
at each decision point, in order, together with what became of that choice, so a
trial that never lands a valid exploit can be characterized directly instead of
inferred from censoring. Nothing here feeds a dependent variable.

Grammar of the ``TrialRow.action_sequence`` cell::

    trace  := "" | token ( "|" token )*
    token  := disposition kind [ "*" esc(note) ] [ ":" target ]
    kind   := "R" (recon) | "E" (exploit) | "S" (stop) | "X" (unrecognized)
    target := esc(service) ":" port ":" esc(endpoint) [ ":" esc(address) ]
                                                          -- kind "E" only
    disposition := ""  executed
                 | "~" honored (a stop: no environment effect, loop ended)
                 | "!" chosen but skipped by the trial-duration guard
                 | "?" chosen but skipped because dispatch did not recognize it
                 | "#" a disposition outside the recorded vocabulary (fail-safe)

Ordinals are **positional**: token *i* is the planner's *i*-th decision, 0-based.
Every ``next_action`` result appends exactly one token on every path, so the
sequence has no gaps and an explicit ordinal would be redundant in a cell that
already has one row per trial.

``note`` is written verbatim (escaped), not as a code: it records why the CLIENT
returned this action when the action is not the model's own choice -- e.g. a stop
coerced from an out-of-range exploit index. That is precisely the distinction the
instrumentation exists to make, so it is spelled out in the cell rather than
compressed into a letter that a later reader would have to look up.

An exploit token carries the **full** ``ServiceTuple`` identity (service, port,
endpoint, address), never the service name alone: ``ServiceTuple`` equality is over
all four fields, and under ``port_rotation`` / ``endpoint_mutation`` /
``network_shuffling`` one service name recurs with different observable identities
inside a single trial -- exactly the conditions the study is about.

**Target-field versioning.** The ARITY of the target IS the version marker: a
3-field target is grammar v1 (written before the address joined ``ServiceTuple``) and
a 4-field target is v2 (:data:`TRACE_GRAMMAR_VERSION`). The decoder accepts both, so
the frozen datasets under ``docs/results/`` -- which hold v1 cells -- keep parsing.
A v1 target decodes with ``address == ""`` and ``DecodedAction.address_recorded ==
False``: the codec reports that the address was never recorded rather than
fabricating one, the same discipline as :data:`NOT_RECORDED` elsewhere. An address
that is genuinely empty still writes its (empty) field, so ``E:svc:80:/p:`` (v2,
empty address) stays distinguishable from ``E:svc:80:/p`` (v1, no address).

``esc`` percent-escapes every reserved byte, so no service identity -- present or
future, however hostile -- can forge a delimiter, and the cell never needs CSV
quoting. Real identities contain none of the reserved characters, so the common
case stays human-readable::

    R|E:http-dvwa:80:/login.php:10.10.0.10|E:ftp-vsftpd:21::10.10.0.14|~S

**Layering.** This module deliberately does NOT import ``attacker``: the attacker
layer depends on ``measurement``, not the reverse. The vocabulary below therefore
restates ``attacker.base``'s constant VALUES (the wire format is the contract), and
``tests/test_action_trace.py`` pins the two vocabularies equal so drift fails loudly
instead of silently encoding a new kind as "unrecognized".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from measurement.models import NOT_RECORDED, ServiceTuple

__all__ = [
    "ACTION_SEP",
    "FIELD_SEP",
    "NOTE_SEP",
    "NOT_RECORDED",
    "TRACE_GRAMMAR_VERSION",
    "DecodedAction",
    "TraceCounts",
    "TraceEntry",
    "count_actions",
    "decode_trace",
    "encode_trace",
    "trace_summary",
]

ACTION_SEP = "|"
FIELD_SEP = ":"
NOTE_SEP = "*"

#: Exploit-target field counts. THE ARITY IS THE VERSION MARKER -- self-describing per
#: token, costing the cell nothing and needing no change to non-exploit tokens. v1 (3
#: fields) predates the address in ``ServiceTuple``; v2 (4 fields) carries it.
_TARGET_ARITY_V1 = 3
_TARGET_ARITY_V2 = 4

#: Which target grammar :func:`encode_trace` writes today (v2 == a 4-field target).
TRACE_GRAMMAR_VERSION = 2

# Action kinds (values mirror ``attacker.base``; pinned by test).
KIND_RECON = "recon"
KIND_EXPLOIT = "exploit"
KIND_STOP = "stop"
KIND_UNKNOWN = "unknown"

# What became of a chosen action (values mirror ``attacker.base``; pinned by test).
DISPOSITION_EXECUTED = "executed"
DISPOSITION_HONORED = "honored"
DISPOSITION_SKIPPED_DURATION_LIMIT = "skipped_duration_limit"
DISPOSITION_SKIPPED_UNRECOGNIZED = "skipped_unrecognized"
#: Fail-safe for a disposition this codec does not know. Never produced while the
#: vocabulary pinning test passes; it exists so an un-mirrored new disposition is
#: recorded as unrecognized rather than silently encoded as "executed".
DISPOSITION_UNRECOGNIZED_CODE = "unrecognized_disposition"

_KIND_CODE: dict[str, str] = {
    KIND_RECON: "R",
    KIND_EXPLOIT: "E",
    KIND_STOP: "S",
    KIND_UNKNOWN: "X",
}
_CODE_KIND: dict[str, str] = {v: k for k, v in _KIND_CODE.items()}

_DISPOSITION_CODE: dict[str, str] = {
    DISPOSITION_EXECUTED: "",
    DISPOSITION_HONORED: "~",
    DISPOSITION_SKIPPED_DURATION_LIMIT: "!",
    DISPOSITION_SKIPPED_UNRECOGNIZED: "?",
}
_CODE_DISPOSITION: dict[str, str] = {v: k for k, v in _DISPOSITION_CODE.items() if v}
_UNKNOWN_DISPOSITION_CODE = "#"
_CODE_DISPOSITION[_UNKNOWN_DISPOSITION_CODE] = DISPOSITION_UNRECOGNIZED_CODE

# Escaped so a token can never be forged, and so the cell never needs CSV quoting:
# every delimiter and prefix of the grammar, plus ',' and '"' (csv), plus every C0/DEL
# control character, plus lone surrogates (see ``_escape``).
_RESERVED = frozenset({"%", ACTION_SEP, FIELD_SEP, NOTE_SEP, ",", '"'} | set(_CODE_DISPOSITION))
_HEX = frozenset(b"0123456789abcdefABCDEF")


class TraceEntry(Protocol):
    """Structural type of one recorded decision (``attacker.base.ActionTraceEntry``)."""

    ordinal: int
    kind: str
    disposition: str
    target: ServiceTuple | None
    note: str


@dataclass(frozen=True)
class DecodedAction:
    """One decision recovered from a trace cell (``ordinal`` is its position)."""

    ordinal: int
    kind: str
    disposition: str
    target: ServiceTuple | None = None
    note: str = ""
    #: False iff this came from a grammar-v1 (3-field) exploit token, i.e. a cell
    #: written before the address was part of ``ServiceTuple``. ``target.address`` is
    #: then the "" DEFAULT, not an observation, and MUST NOT be compared for identity.
    address_recorded: bool = True

    @property
    def executed(self) -> bool:
        """True iff the action was dispatched to the environment."""

        return self.disposition == DISPOSITION_EXECUTED


def _escape(text: str) -> str:
    """Percent-escape reserved and control characters (``%`` itself -> ``%25``).

    Lone surrogates are escaped too: they are reachable (``json.loads`` of an ingested
    log can yield one, and ``measurement.collectors.parsers`` builds a ``ServiceTuple``
    from such JSON) and are not encodable as UTF-8, so an unescaped one would make
    ``csv.writer`` raise at the very END of a run and destroy the dataset.
    """

    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch in _RESERVED or code < 0x20 or code == 0x7F or 0xD800 <= code <= 0xDFFF:
            out.extend(f"%{b:02X}" for b in ch.encode("utf-8", "surrogatepass"))
        else:
            out.append(ch)
    return "".join(out)


def _unescape(text: str) -> str:
    """Inverse of :func:`_escape`. A stray ``%`` stays literal, never an error."""

    if "%" not in text:
        return text
    raw = text.encode("utf-8", "surrogatepass")
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == 0x25 and i + 2 < len(raw) and raw[i + 1] in _HEX and raw[i + 2] in _HEX:
            out.append(int(raw[i + 1 : i + 3], 16))
            i += 3
        else:
            out.append(raw[i])
            i += 1
    return out.decode("utf-8", errors="surrogatepass")


def _encode_target(target: ServiceTuple) -> str:
    return (
        f"{_escape(target.service)}{FIELD_SEP}{target.port}{FIELD_SEP}"
        f"{_escape(target.endpoint)}{FIELD_SEP}{_escape(target.address)}"
    )


def _encode_action(entry: TraceEntry) -> str:
    code = _KIND_CODE.get(entry.kind, _KIND_CODE[KIND_UNKNOWN])
    prefix = _DISPOSITION_CODE.get(entry.disposition, _UNKNOWN_DISPOSITION_CODE)
    note = getattr(entry, "note", "") or ""
    token = f"{prefix}{code}"
    if note:
        token += f"{NOTE_SEP}{_escape(note)}"
    target = getattr(entry, "target", None)
    if code == "E" and target is not None:
        token += f"{FIELD_SEP}{_encode_target(target)}"
    return token


def encode_trace(entries: Sequence[TraceEntry]) -> str:
    """Encode a decision sequence to the one-cell trace string.

    An empty sequence encodes to ``""``, which is NOT :data:`NOT_RECORDED`: "the
    planner made no decisions" stays distinguishable from "no trace was captured".
    """

    return ACTION_SEP.join(_encode_action(e) for e in entries)


def decode_trace(text: str) -> tuple[DecodedAction, ...]:
    """Decode a trace cell back into :class:`DecodedAction` records.

    Inverse of :func:`encode_trace` for ``kind``, ``disposition``, ``note`` and (for
    exploits) ``target``; ``ordinal`` comes back from the token's position, which is
    what the encoder relied on. A target attached to a NON-exploit kind is not carried
    in the cell, because only ``E`` tokens have a target field.

    Raises ``ValueError`` on a malformed cell (including :data:`NOT_RECORDED`) so a
    tool pointed at the wrong column fails loudly; the reporting path uses
    :func:`trace_summary`, which degrades to ``{"note": ...}`` instead.
    """

    if text == NOT_RECORDED:
        raise ValueError("action_sequence was not recorded for this row")
    if text == "":
        return ()
    out: list[DecodedAction] = []
    for ordinal, token in enumerate(text.split(ACTION_SEP)):
        if not token:
            raise ValueError(f"empty trace token at position {ordinal}")
        disposition = DISPOSITION_EXECUTED
        body = token
        if token[0] in _CODE_DISPOSITION:
            disposition = _CODE_DISPOSITION[token[0]]
            body = token[1:]
        parts = body.split(FIELD_SEP)
        head = parts[0]
        code = head[:1]
        if code not in _CODE_KIND:
            raise ValueError(f"unknown action code {code!r} in trace token {token!r}")
        note = ""
        if len(head) > 1:
            if head[1] != NOTE_SEP:
                raise ValueError(f"malformed trace token {token!r}: expected {NOTE_SEP!r}")
            note = _unescape(head[2:])
        target: ServiceTuple | None = None
        address_recorded = True
        if code == "E" and len(parts) != 1:
            # A bare "E" (exploit with no target) is legal and decodes to target=None:
            # a codec in a study artifact must never write a cell it cannot read.
            if len(parts) not in (_TARGET_ARITY_V1 + 1, _TARGET_ARITY_V2 + 1):
                raise ValueError(f"malformed exploit token {token!r}: expected 4 or 5 fields")
            try:
                port = int(parts[2])
            except ValueError as exc:
                raise ValueError(f"malformed port in trace token {token!r}") from exc
            address_recorded = len(parts) == _TARGET_ARITY_V2 + 1
            address = _unescape(parts[4]) if address_recorded else ""
            target = ServiceTuple(_unescape(parts[1]), port, _unescape(parts[3]), address)
        elif code != "E" and len(parts) != 1:
            raise ValueError(f"malformed token {token!r}: {code!r} takes no target")
        out.append(
            DecodedAction(
                ordinal=ordinal,
                kind=_CODE_KIND[code],
                disposition=disposition,
                target=target,
                note=note,
                address_recorded=address_recorded,
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class TraceCounts:
    """Decision counts derived from a trace (see :func:`count_actions`)."""

    n_actions: int
    n_recon_actions: int
    n_exploit_actions: int


def count_actions(entries: Sequence[TraceEntry]) -> TraceCounts:
    """Count DECISIONS -- what the planner chose, carried out or not.

    ``n_actions`` is the number of trace tokens: the number of recorded decisions.
    ``n_recon_actions`` / ``n_exploit_actions`` count recon / exploit decisions
    *including* any the guards skipped. Counting decisions rather than executions is
    what makes these columns new information -- the executed counts are already
    recoverable as ``len(recon_snapshots)`` / ``len(exploit_deliveries)``, and from
    the trace itself by ignoring the skipped dispositions.
    """

    return TraceCounts(
        n_actions=len(entries),
        n_recon_actions=sum(1 for e in entries if e.kind == KIND_RECON),
        n_exploit_actions=sum(1 for e in entries if e.kind == KIND_EXPLOIT),
    )


def trace_summary(text: object) -> dict:
    """Descriptive summary of one trace cell; degrades instead of raising.

    Accepts ``object`` on purpose: ``analysis.dataset.load_csv`` renders an empty
    trace cell as a float ``NaN``, so a summariser that assumed ``str`` would raise
    ``AttributeError`` on exactly the zero-decision rows this instrumentation exists
    to characterize. Non-strings degrade to a ``{"note": ...}`` form.
    """

    if not isinstance(text, str):
        return {"note": f"action_sequence is not a string: {type(text).__name__}"}
    if text == NOT_RECORDED:
        return {"note": "action_sequence not recorded for this row"}
    try:
        actions = decode_trace(text)
    except ValueError as exc:
        return {"note": f"unparseable action_sequence: {exc}"}
    counts = count_actions(actions)
    return {
        "n_actions": counts.n_actions,
        "n_recon_actions": counts.n_recon_actions,
        "n_exploit_actions": counts.n_exploit_actions,
        "n_stop_actions": sum(1 for a in actions if a.kind == KIND_STOP),
        "n_unknown_actions": sum(1 for a in actions if a.kind == KIND_UNKNOWN),
        "n_not_executed": sum(1 for a in actions if not a.executed),
        "n_client_coerced": sum(1 for a in actions if a.note),
        # The stops alone: from prompt format 2 the client also forces a SCAN when the model asks
        # to exploit or stop before its attempt's first scan (``coerced_recon_before_scan``).
        "n_client_coerced_stops": sum(1 for a in actions if a.note and a.kind == KIND_STOP),
    }
