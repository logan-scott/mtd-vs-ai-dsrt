"""LLM client backends for the attacker planner.

- ``StubLLMClient`` — a deterministic offline policy (recon, then exploit the
  discovered services it has not already tried and has not learned are decoys,
  stop on first genuine compromise). Requires no API key and makes datasets
  reproducible for tests and CI.
- ``AnthropicClient`` — the real Claude-backed planner (the primary attacker
  model). It calls the Anthropic Messages API with a recon/exploit/stop tool set
  and parses the chosen tool call into an action. The tools still resolve against
  the offline mock target — the model plans, but no live testbed network is touched.
  The stable request prefix (system prompt + tool definitions) is marked with
  ``cache_control`` so repeated calls within a trial hit the prompt cache, and the
  per-response token usage breakdown is accumulated on the client so the cost-probe
  can read real cached-vs-uncached counts.
- ``OpenAIClient`` / ``DeepSeekClient`` — staged scaffolds for the later
  supplementary robustness arms (see ``build_attacker``). They conform to the
  ``LLMClient`` interface but their call bodies are not implemented yet.

``build_attacker(config)`` selects the backend from the experiment config.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from attacker.base import (
    CLOCK_MODELED,
    NOTE_COERCED_RECON_BEFORE_SCAN,
    NOTE_COERCED_RECON_NO_TOOL_USE,
    NOTE_COERCED_STOP_INDEX_OUT_OF_RANGE,
    NOTE_COERCED_STOP_NO_TOOL_USE,
    NOTE_NONE,
    AttemptMemory,
    observed_outcome,
    tool_choice_payload,
)
from attacker.errors import AttackerApiError
from attacker.planner import (
    Action,
    ExploitAction,
    LLMAttacker,
    PlannerState,
    ReconAction,
    StopAction,
)
from measurement.models import ServiceTuple

_DEFAULT_PROMPT = "You are an autonomous offensive workflow operating only in an isolated lab."

# Sentinel rendered in the discovered-services listing for a service whose modeled
# address is "" (the ``ServiceTuple.address`` default — e.g. a v1 action-trace token
# decoded without one). A missing address renders as unknown, never as a fabricated
# guess. Public so tests and docs can pin the literal rather than repeat it.
ADDRESS_UNKNOWN = "-"

_EPHEMERAL = {"type": "ephemeral"}

# Usage-breakdown fields threaded out of each response (Anthropic Messages API).
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


@dataclass
class StubLLMClient:
    """Deterministic offline planning policy (no model calls)."""

    def next_action(self, state: PlannerState, system_prompt: str) -> Action:
        if not state.reconned:
            return ReconAction()
        if state.compromised_genuine:
            return StopAction()
        candidates = sorted(state.discovered - state.attempted - state.known_decoys)
        if not candidates:
            return StopAction()
        return ExploitAction(candidates[0])


def _extract_usage(resp) -> dict[str, int]:
    """Pull the token-usage breakdown from a response, defaulting missing fields to 0.

    Absent/None fields (e.g. when prompt caching is unavailable) degrade to 0 so the
    accumulated totals stay well-formed.
    """

    usage = getattr(resp, "usage", None)
    out: dict[str, int] = {}
    for name in _USAGE_FIELDS:
        value = getattr(usage, name, None) if usage is not None else None
        out[name] = int(value) if value is not None else 0
    return out


@dataclass
class AnthropicClient:
    """Claude-backed planner (calls the Anthropic Messages API).

    Resilience (operational only — never affects sampled outputs): the SDK client is
    created with a configurable ``max_retries`` and ``timeout`` (the SDK uses
    exponential backoff and honors ``Retry-After``). A thin application-level wrapper
    adds a second layer of backoff-with-jitter around ``messages.create`` for the
    transient error family (overload / rate-limit / connection / timeout) so a brief
    outage window longer than the SDK budget does not kill a multi-hour run; on final
    exhaustion it raises the typed ``AttackerApiError`` for the runner to classify.
    ``sleep``/``jitter`` are injectable so the backoff is unit-tested without real waits.
    """

    model: str
    # None == omit ``temperature`` from the request entirely, so the model applies its
    # own default. Claude 4.7+ (e.g. claude-sonnet-5) REJECTS an explicitly-supplied
    # temperature/top_p/top_k with a 400, so those models can only be driven with this
    # unset. A number is sent exactly as before. See ``_request_kwargs``.
    temperature: float | None = 0.0
    # None == omit ``tool_choice`` from the request entirely (the API default, and what
    # every frozen arm sent). "any" forces a tool call, "auto" sends the default
    # explicitly. This is an experimental manipulation, not an operational knob -- see
    # ``attacker.base`` for why it is a three-state string rather than a bool.
    tool_choice: str | None = None
    max_tokens: int = 512
    api_key: str | None = None
    cache_prompt: bool = True
    # Resilience knobs (see docstring). SDK-level:
    max_retries: int = 8
    timeout: float = 120.0
    # Application-level backoff (beyond the SDK's own retries):
    api_retry_attempts: int = 6
    api_retry_base_s: float = 2.0
    api_retry_cap_s: float = 60.0
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)
    jitter: Callable[[], float] = field(default=random.random, repr=False, compare=False)
    _client: object | None = field(default=None, repr=False, compare=False)
    # Per-response usage records accumulated across calls (read by the cost-probe).
    usage_records: list[dict] = field(default_factory=list, repr=False, compare=False)
    # Instrumentation only: why the LAST returned action is what it is. Set on every
    # return path of ``next_action``; read by the planner's action trace. "" means the
    # model's own tool call was returned unchanged.
    last_action_note: str = field(default=NOTE_NONE, repr=False, compare=False)
    # Instrumentation only: how many APP-LEVEL retries the last ``messages.create`` needed
    # (0 == first try succeeded). Read by the planner's action timing so a planner call whose
    # latency includes backoff sleeps can be told apart from a slow model. The SDK's own
    # retries are internal to it and are visible only as latency.
    last_call_retries: int = field(default=0, repr=False, compare=False)
    # Provenance only: how many responses the API returned per served model string (the
    # response's ``model`` field), so a run records which snapshot actually answered (C235).
    served_models: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def _ensure(self):
        if self._client is None:
            import anthropic  # imported lazily so the SDK is optional

            self._client = anthropic.Anthropic(
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._client

    @staticmethod
    def _retryable_exceptions() -> tuple[type[BaseException], ...]:
        """The transient API error family the app-level wrapper retries."""

        import anthropic

        return (
            anthropic.OverloadedError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,  # 5xx (e.g. 500/503) — transient server-side
        )

    def _create_message(self, **kwargs):
        """Call ``messages.create`` with app-level backoff+jitter over the SDK retries.

        Retries the transient error family up to ``api_retry_attempts`` times with
        exponential backoff (base ``api_retry_base_s``, cap ``api_retry_cap_s``) plus
        jitter; on exhaustion raises ``AttackerApiError``. Non-transient errors (e.g.
        auth, bad request, a bug) propagate immediately — they are not retried."""

        client = self._ensure()
        retryable = self._retryable_exceptions()
        attempt = 0
        while True:
            try:
                response = client.messages.create(**kwargs)
            except retryable as exc:
                attempt += 1
                if attempt >= self.api_retry_attempts:
                    raise AttackerApiError(
                        f"Anthropic API unrecoverable after {attempt} app-level retries "
                        f"(SDK max_retries={self.max_retries}, timeout={self.timeout}s): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                base = min(self.api_retry_cap_s, self.api_retry_base_s * (2 ** (attempt - 1)))
                # Jitter in [0.5, 1.0]*base so concurrent workers desynchronize.
                self.sleep(base * (0.5 + 0.5 * self.jitter()))
            else:
                self.last_call_retries = attempt
                return response

    def reset_usage(self) -> None:
        """Clear accumulated usage records (call before profiling a fresh trial)."""

        self.usage_records = []

    @property
    def call_count(self) -> int:
        return len(self.usage_records)

    def usage_totals(self) -> dict[str, int]:
        """Sum the usage breakdown across accumulated calls."""

        totals = {name: 0 for name in _USAGE_FIELDS}
        for rec in self.usage_records:
            for name in _USAGE_FIELDS:
                totals[name] += int(rec.get(name, 0))
        return totals

    def _tools(self) -> list[dict]:
        empty = {"type": "object", "properties": {}, "required": []}
        tools = [
            {
                "name": "recon",
                "description": "Scan the perimeter to (re)discover reachable services.",
                "input_schema": empty,
            },
            {
                "name": "exploit",
                "description": "Deliver an exploit to a discovered service by its list index.",
                "input_schema": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            },
            {"name": "stop", "description": "Stop this attempt.", "input_schema": empty},
        ]
        # Mark the last tool so the whole tool-definition prefix is cached. Tools
        # render before the system prompt, so this breakpoint caches every tool.
        if self.cache_prompt and tools:
            tools[-1] = {**tools[-1], "cache_control": dict(_EPHEMERAL)}
        return tools

    def _system(self, system_prompt: str):
        """Wrap the system prompt as a cache-marked content block (or plain string)."""

        if not self.cache_prompt:
            return system_prompt
        return [{"type": "text", "text": system_prompt, "cache_control": dict(_EPHEMERAL)}]

    def _request_kwargs(self, system_prompt: str, user: str) -> dict:
        """Build the exact ``messages.create`` kwargs, omitting unset sampling params.

        ``temperature=None`` is NOT passed through as an explicit null -- the key is left
        out of the payload altogether. Claude 4.7+ rejects the parameter's mere presence
        (``400: `temperature` is deprecated for this model``), so sending ``None`` would
        fail exactly as sending ``0.7`` does. Omission is the only accepted form.

        When a number is configured the kwargs are byte-identical to what this client sent
        before the parameter became optional, so the frozen Haiku 4.5 arm is unaffected.

        ``tool_choice`` follows the same discipline for the same reason: ``None`` leaves
        the key out of the payload, so the DEFAULT request -- key set AND key order -- is
        byte-identical to the one every archived run issued. That identity is what makes
        the setting usable as an experimental manipulation instead of a fourth uncontrolled
        difference between arms, so it is pinned by test against a hardcoded literal.
        """

        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens}
        if self.temperature is not None:
            # Inserted here, not appended, so a configured temperature reproduces the
            # pre-change key order exactly as well as the pre-change key set.
            kwargs["temperature"] = self.temperature
        kwargs["system"] = self._system(system_prompt)
        kwargs["tools"] = self._tools()
        choice = tool_choice_payload(self.tool_choice)
        if choice is not None:
            # Next to ``tools``, which it modifies. Unset -> the key never appears.
            kwargs["tool_choice"] = choice
        kwargs["messages"] = [{"role": "user", "content": user}]
        return kwargs

    @staticmethod
    def render_service_line(index: int, service: ServiceTuple) -> str:
        """Render one discovered service for the planner prompt.

        THE FORMAT (stable, and documented in ``docs/metrics_specification.md``)::

            {index}: {service} address={address} port={port} path={endpoint}

        Every attribute the ground-truth ``ServiceTuple`` distinguishes is rendered, so
        all four evaluable observable attributes are perceivable by the agent under
        test. Rendering only three of them is what made ``network_shuffling`` mutate an
        attribute invisible to the planner while ``port_rotation`` and
        ``endpoint_mutation`` mutated visible ones — an asymmetry across the RQ2
        technique contrast (``docs/null_condition_audit.md``, Finding 1).

        Two field-order properties are load-bearing and must not be "tidied":

        * ``address=`` is placed BEFORE ``path=``. ``path`` is the only free-form field
          (an endpoint is an arbitrary string), so it goes LAST: any ``key=value`` field
          rendered after it could be forged by an endpoint containing whitespace. With
          ``path`` last, an endpoint may contain ``=`` or spaces without any earlier
          field becoming ambiguous. The address itself is a bounded dotted-quad token,
          so it cannot swallow the fields around it.
        * ``address`` / ``port`` / ``path`` are all ``key=value``, so the reader never
          has to infer a field from its position, and ``service address=… port=…`` reads
          in natural network order.

        The rendered address is the MODELED OBSERVABLE address (10.10.x.x), i.e. what
        the attacker fingerprints. It is never the physical delivery host: delivery and
        the ``harness.safety.assert_within_cidr`` allowlist use
        ``harness.live_targets.LiveService.delivery_host()`` alone.
        """

        return f"{index}: {AnthropicClient.render_identity(service)}"

    @staticmethod
    def render_identity(service: ServiceTuple) -> str:
        """One service's four-field identity, exactly as the listing shows it (no ordinal).

        ``{service} address={address} port={port} path={endpoint}`` -- the listing line without
        its ``{index}: `` prefix. Outcome lines and the cross-attempt memory use the same
        rendering, so a service whose address, port or path changed reads as a different
        service, not under its old result. ``path`` stays last (it is free-form).
        """

        return (
            f"{service.service}"
            f" address={service.address or ADDRESS_UNKNOWN}"
            f" port={service.port}"
            f" path={service.endpoint or '/'}"
        )

    @staticmethod
    def render_outcome_line(target: ServiceTuple, outcome: str) -> str:
        """``- {outcome}: {identity}`` (outcome first: the identity's free-form path ends the line)."""

        return f"- {outcome}: {AnthropicClient.render_identity(target)}"

    @staticmethod
    def render_memory(memory: tuple[AttemptMemory, ...]) -> str:
        """The cross-attempt memory block ("" on a sequence's first attempt).

        For every earlier attempt: how many scans returned, and each exploit sent with its
        result, by full identity, in order. Then the services the
        previous attempt's last scan listed -- the context the design specifies ("the
        services discovered on the previous attempt"). No ordinals: only the current listing
        can be exploited by index.
        """

        if not memory:
            return ""
        lines = ["Earlier attempts in this sequence:"]
        for m in memory:
            scans = (
                "no scan" if m.scans == 0 else ("1 scan" if m.scans == 1 else f"{m.scans} scans")
            )
            if m.exploits:
                lines.append(f"Attempt {m.attempt_index}: {scans}. Exploit outcomes:")
                lines.extend(AnthropicClient.render_outcome_line(t, o) for t, o in m.exploits)
            else:
                lines.append(f"Attempt {m.attempt_index}: {scans}. Exploit outcomes: none")
        last = memory[-1]
        if last.last_listing:
            lines.append(f"Services listed by the last scan of attempt {last.attempt_index}:")
            lines.extend(f"- {AnthropicClient.render_identity(t)}" for t in last.last_listing)
        else:
            lines.append(f"Services listed by the last scan of attempt {last.attempt_index}: none")
        return "\n".join(lines)

    def next_action(self, state: PlannerState, system_prompt: str) -> Action:
        # Ordered by SERVICE first: ``ServiceTuple`` is ``order=True`` with ``address``
        # LAST, so the address only ever breaks a tie among identical
        # service/port/endpoint triples. The ordinal the model selects therefore resolves
        # against the same list the listing was rendered from, line for line.
        services = sorted(state.discovered)
        listing = "\n".join(self.render_service_line(i, s) for i, s in enumerate(services))
        # The attempt's earlier exploit outcomes, keyed by the FULL identity (format 2). Until
        # 2026-10-31 this line was keyed on ``t.service`` alone, so after a rescan a service that
        # had moved still read under its old outcome (usually "unreachable"); that discouraged a
        # return to a moved genuine service in the conditions that move services, and kept a
        # moved decoy reading "reached" -- a property of the instrument, not of the defense.
        # Rendered exactly like the listing, a moved service now reads as one not yet tried.
        prior_lines = [
            self.render_outcome_line(
                t, observed_outcome(o.reached_and_executed, o.unauthorized_access)
            )
            for t, o in state.outcomes
        ]
        prior = "\n" + "\n".join(prior_lines) if prior_lines else " none"
        memory = self.render_memory(state.memory)
        user = (
            f"Attempt {state.attempt_index}. Actions used {state.actions_used}/{state.max_actions}.\n"
            f"Reconned: {state.reconned}. Genuine service compromised: {state.compromised_genuine}.\n"
            f"Discovered services:\n{listing or '(none yet)'}\n"
            f"Prior exploit outcomes this attempt:{prior}\n"
            + (f"{memory}\n" if memory else "")
            + "Choose the single next action via a tool call."
        )
        resp = self._create_message(**self._request_kwargs(system_prompt, user))
        self.usage_records.append(_extract_usage(resp))
        served = getattr(resp, "model", None)
        if isinstance(served, str) and served:
            self.served_models[served] = self.served_models.get(served, 0) + 1
        # Reset before the branches so a stale note can never leak into a later entry.
        self.last_action_note = NOTE_NONE
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                if block.name == "recon":
                    return ReconAction()
                if not state.reconned and block.name in ("stop", "exploit"):
                    # Before the attempt's first scan there is no listing, and the workflow's
                    # first step is a scan: run it (``NOTE_COERCED_RECON_BEFORE_SCAN``).
                    self.last_action_note = NOTE_COERCED_RECON_BEFORE_SCAN
                    return ReconAction()
                if block.name == "stop":
                    return StopAction()
                if block.name == "exploit":
                    idx = int(block.input.get("index", 0))
                    if 0 <= idx < len(services):
                        return ExploitAction(services[idx])
                    # The model asked to exploit a service that is not in the listing.
                    # This has always been coerced into a stop; the note is what makes
                    # that coercion visible instead of looking like a chosen stop.
                    self.last_action_note = NOTE_COERCED_STOP_INDEX_OUT_OF_RANGE
                    return StopAction()
        # No tool_use block at all: the response did not honor the tool contract.
        if not state.reconned:
            self.last_action_note = NOTE_COERCED_RECON_NO_TOOL_USE
            return ReconAction()
        self.last_action_note = NOTE_COERCED_STOP_NO_TOOL_USE
        return StopAction()


# --------------------------------------------------------------------------- #
# Staged robustness-arm adapters (not yet implemented)
# --------------------------------------------------------------------------- #


def _openai_style_tools() -> list[dict]:
    """The recon/exploit/stop tools in OpenAI/DeepSeek function-calling format.

    OpenAI's chat-completions API nests each tool under ``{"type": "function",
    "function": {...}}`` with a ``parameters`` JSON schema (vs. Anthropic's
    top-level ``input_schema``). Provided so the staged adapters have a concrete
    tool surface to build on; the call bodies remain to be implemented.
    """

    empty = {"type": "object", "properties": {}, "required": []}
    specs = [
        ("recon", "Scan the perimeter to (re)discover reachable services.", empty),
        (
            "exploit",
            "Deliver an exploit to a discovered service by its list index.",
            {"type": "object", "properties": {"index": {"type": "integer"}}, "required": ["index"]},
        ),
        ("stop", "Stop this attempt.", empty),
    ]
    return [
        {"type": "function", "function": {"name": n, "description": d, "parameters": p}}
        for n, d, p in specs
    ]


@dataclass
class OpenAIClient:
    """Staged OpenAI-backed planner (supplementary robustness arm — not built).

    Mirrors ``AnthropicClient``: it would map the recon/exploit/stop tools to
    OpenAI's function-calling format (see ``_openai_style_tools``), issue a
    chat-completions request, and parse the chosen ``tool_call`` into an
    ``Action``. Wired into ``build_attacker`` so the backend name is recognized,
    but the call body is deferred to the later robustness subsets.
    """

    model: str
    temperature: float | None = 0.0
    max_tokens: int = 512
    api_key: str | None = None
    base_url: str | None = None

    @staticmethod
    def _tools() -> list[dict]:
        return _openai_style_tools()

    def next_action(self, state: PlannerState, system_prompt: str) -> Action:
        raise NotImplementedError("staged: implement for the supplementary robustness arms")


@dataclass
class DeepSeekClient:
    """Staged DeepSeek-backed planner (supplementary robustness arm — not built).

    DeepSeek is OpenAI-API-compatible, so this adapter follows the same
    function-calling shape as ``OpenAIClient`` and can point ``base_url`` at a
    self-hosted open-weight endpoint later. Wired into ``build_attacker`` but the
    call body is deferred to the later robustness subsets.
    """

    model: str
    temperature: float | None = 0.0
    max_tokens: int = 512
    api_key: str | None = None
    base_url: str | None = None

    @staticmethod
    def _tools() -> list[dict]:
        return _openai_style_tools()

    def next_action(self, state: PlannerState, system_prompt: str) -> Action:
        raise NotImplementedError("staged: implement for the supplementary robustness arms")


def _load_prompt(config) -> str:
    path = config.attacker.system_prompt_path
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return _DEFAULT_PROMPT


#: The example config's ``attacker.model_version`` placeholder. Every archived run recorded it
#: verbatim in ``attacker_model_version`` (C235); the pinned, dated ``model_id`` IS the version.
MODEL_VERSION_PLACEHOLDER = "<pin-at-runtime>"


def _resolved_model_version(attacker_params) -> str:
    """The version string recorded per row: the configured one, or the pinned model ID when the
    config still carries the placeholder (a dated snapshot ID names the exact model)."""

    version = str(getattr(attacker_params, "model_version", "") or "")
    if version in ("", MODEL_VERSION_PLACEHOLDER):
        return str(attacker_params.model_id)
    return version


def build_attacker(config, backend: str | None = None):
    """Instantiate the attacker for the configured (or overridden) backend.

    The planner backends (``stub``/``anthropic``/...) honor ``attacker.clock`` and
    ``attacker.record_timing``. The offline ``mock`` attacker draws its own modeled timings and
    performs no real actions, so there is nothing for a wall clock to measure: asking for
    ``clock: wall`` with it raises rather than silently running the modeled clock, and
    ``record_timing`` is a no-op for it (its rows read the timing columns as not recorded).
    """

    from attacker.mock import MockAttacker

    backend = backend or config.attacker.backend
    clock_mode = getattr(config.attacker, "clock", CLOCK_MODELED)
    if backend == "mock":
        if clock_mode != CLOCK_MODELED:
            raise ValueError(
                f"attacker.clock={clock_mode!r} needs the planner (backend 'stub' or "
                "'anthropic'); the offline 'mock' attacker has no real actions to time"
            )
        return MockAttacker()
    prompt = _load_prompt(config)
    if backend == "stub":
        client: object = StubLLMClient()
    elif backend == "anthropic":
        client = AnthropicClient(
            model=config.attacker.model_id,
            temperature=config.attacker.temperature,
            tool_choice=config.attacker.tool_choice,
            max_retries=config.attacker.max_retries,
            timeout=config.attacker.request_timeout_s,
            api_retry_attempts=config.attacker.api_retry_attempts,
        )
    elif backend == "openai":
        client = OpenAIClient(
            model=config.attacker.model_id, temperature=config.attacker.temperature
        )
    elif backend == "deepseek":
        client = DeepSeekClient(
            model=config.attacker.model_id, temperature=config.attacker.temperature
        )
    else:
        raise ValueError(f"unknown attacker backend: {backend!r}")
    return LLMAttacker(
        client=client,
        model_id=config.attacker.model_id,
        model_version=_resolved_model_version(config.attacker),
        system_prompt=prompt,
        clock_mode=clock_mode,
        record_timing=bool(getattr(config.attacker, "record_timing", False)),
    )
