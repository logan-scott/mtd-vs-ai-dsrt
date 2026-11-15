"""MTD techniques as deterministic configuration-state transforms.

Each technique is a pure function ``(state, rng) -> (new_state, changed_ids)``:
given a configuration state and a seeded RNG, it returns the reconfigured state
and the ids of services whose observable attributes changed. Determinism comes
entirely from the RNG the orchestrator supplies (identical seed → identical
mutations), which is what makes the whole pipeline reproducible.

Techniques mutate only their own attribute class so their effects are separable:

- ``static_baseline``      — identity (comparator; no mutation).
- ``network_shuffling``    — reassign genuine services' IP bindings.
- ``port_rotation``        — reassign genuine services' ports.
- ``endpoint_mutation``    — regenerate genuine services' endpoints/paths.
- ``deceptive_presentation`` — refresh decoys' observable identities.
- ``combined_mtd_deception`` — one or more mutation techniques + deception.

Only genuine services are mutated by the mutation techniques; deception acts on
decoys. Genuine services' stable identity and vulnerability profile are never
touched (so ``profile_checksum`` is invariant — see ``mtd/checksums.py``).

**Guaranteed-distinct redraws (``_mutate_genuine(..., distinct=True)``).** A generator
can by chance redraw a service's CURRENT value, which is a silent no-op for that
service on that tick and understates churn. ``network_shuffling`` resamples until the
value differs; ``port_rotation`` and ``endpoint_mutation`` deliberately do NOT. The
asymmetry is intentional and quantified:

- Rates: address 1/65,024 = 1.5e-5 per draw; port 1/64,512 = 1.6e-5; endpoint
  1/16,777,216 = 6.0e-8. Measured over a full-design offline replay (20 sequences x 8
  attempts x 6 conditions, cadence 35 s, horizon 1800 s): 1 collision in 41,170 address
  draws, 1 in 82,225 port draws, 0 in 82,285 endpoint draws.
- A resample consumes an EXTRA draw from the technique's RNG, so switching
  ``port_rotation``/``endpoint_mutation`` to ``distinct=True`` would re-align every
  later draw in the stream. Because a real port collision occurs at full design scale,
  that would move already-collected port_rotation outcomes. The archived datasets under
  ``docs/results/`` must stay reproducible, so those two arms keep the original path.
- Address is different: the whole condition rides on one attribute, and the range is the
  one most likely to be narrowed later (re-pointing ``_new_ip`` at a /24 would raise the
  rate ~256x to 1/254).

A no-op tick for ALL genuine services at once — the only way a whole reconfiguration
event could be fabricated — has probability (1/65,024)^5 < 9e-25 for the address (exactly
8.60e-25) and was never observed.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from mtd.models import ConfigState, ServiceConfig

# A technique: (state, rng) -> (new_state, changed_service_ids)
Technique = Callable[[ConfigState, random.Random], tuple[ConfigState, tuple[str, ...]]]

#: Resample budget for ``distinct=True``. Bounded so the transform always terminates;
#: with the real generators eight consecutive collisions is a < 4e-39 event (exactly
#: (1/65,024)**8 = 3.13e-39), and on (impossible) exhaustion the last draw is kept rather
#: than raising — a study run must not abort on a cosmic-ray-probability event.
_MAX_DISTINCT_RESAMPLES = 8

# Default sub-techniques composed by the combined condition.
COMBINED_MEMBERS: tuple[str, ...] = (
    "endpoint_mutation",
    "port_rotation",
    "deceptive_presentation",
)


# --------------------------------------------------------------------------- #
# Attribute generators (deterministic given the RNG)
# --------------------------------------------------------------------------- #


def _new_ip(rng: random.Random) -> str:
    """A modeled address in the 10.10.0.0/16 space the initial profile is laid out in.

    Range: ``10.10.[0-255].[1-254]`` = 65,024 distinct values, and every address in
    ``testbed/profile.py`` (genuine ``10.10.0.10``-``.14``, decoys ``10.10.0.90``-``.92``)
    lies inside it, so a shuffled service moves WITHIN the modeled address family rather
    than to a different-looking one. This is the MODELED, attacker-observable address
    only: it is never a delivery target and must never be fed to
    ``harness.safety.assert_within_cidr`` (the live allowlist is ``172.31.66.0/24``, so
    asserting a modeled address there would abort every live run). Delivery and the
    allowlist use the live layer's PHYSICAL delivery address; ``harness/live_targets.py``
    owns which field that is — never this one.
    """

    return f"10.10.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _new_port(rng: random.Random) -> int:
    return rng.randint(1024, 65535)


def _new_endpoint(rng: random.Random) -> str:
    return f"/api/{rng.getrandbits(24):06x}"


# --------------------------------------------------------------------------- #
# Techniques
# --------------------------------------------------------------------------- #


def static_baseline(state: ConfigState, rng: random.Random) -> tuple[ConfigState, tuple[str, ...]]:
    return state, ()


def _mutate_genuine(
    state: ConfigState,
    rng: random.Random,
    attr: str,
    gen: Callable[[random.Random], object],
    *,
    distinct: bool = False,
) -> tuple[ConfigState, tuple[str, ...]]:
    """Reassign one observable attribute on every genuine service.

    ``distinct=True`` resamples until the drawn value differs from the service's
    CURRENT value, so the reassignment is guaranteed to be a real change. It is OFF
    by default and that default is LOAD-BEARING: a resample consumes an extra draw,
    which shifts every subsequent value the technique draws from the shared RNG. Only
    ``network_shuffling`` opts in — see the module docstring.
    """

    services: list[ServiceConfig] = []
    changed: list[str] = []
    for s in state.services:
        if s.genuine:
            value = gen(rng)
            if distinct:
                current = getattr(s, attr)
                tries = 0
                while value == current and tries < _MAX_DISTINCT_RESAMPLES:
                    value = gen(rng)
                    tries += 1
            services.append(s.with_attrs(**{attr: value}))
            changed.append(s.service_id)
        else:
            services.append(s)
    return state.with_services(tuple(services)), tuple(changed)


def network_shuffling(
    state: ConfigState, rng: random.Random
) -> tuple[ConfigState, tuple[str, ...]]:
    """Reassign every genuine service's observable ADDRESS (guaranteed to change).

    ``distinct=True`` because the address is this condition's ONLY manipulated
    attribute: a redrawn-identical address would be a silent no-op tick for that
    service, understating churn on the one signal the condition carries. Port and
    endpoint rotation each mutate one of several correlated observables and are
    deliberately left on the shared (non-resampling) path — see the module docstring.
    """

    return _mutate_genuine(state, rng, "ip", _new_ip, distinct=True)


def port_rotation(state: ConfigState, rng: random.Random) -> tuple[ConfigState, tuple[str, ...]]:
    return _mutate_genuine(state, rng, "port", _new_port)


def endpoint_mutation(
    state: ConfigState, rng: random.Random
) -> tuple[ConfigState, tuple[str, ...]]:
    return _mutate_genuine(state, rng, "endpoint", _new_endpoint)


def deceptive_presentation(
    state: ConfigState, rng: random.Random
) -> tuple[ConfigState, tuple[str, ...]]:
    """Refresh decoys' observable identities (rotate fake ports/endpoints).

    Deception acts only on decoys; genuine services are untouched. Requires that
    the initial state already seeds decoys — ``default_profile`` does so for every
    condition, not just this one, so what distinguishes this technique is that it
    ROTATES the decoys rather than that it introduces them. Each firing rotates
    decoy port + endpoint so the deceptive surface shifts.
    """

    services: list[ServiceConfig] = []
    changed: list[str] = []
    for s in state.services:
        if not s.genuine:
            services.append(s.with_attrs(port=_new_port(rng), endpoint=_new_endpoint(rng)))
            changed.append(s.service_id)
        else:
            services.append(s)
    return state.with_services(tuple(services)), tuple(changed)


def make_combined(members: tuple[str, ...] = COMBINED_MEMBERS) -> Technique:
    """Build a combined technique that applies ``members`` in order, one RNG."""

    def combined(state: ConfigState, rng: random.Random) -> tuple[ConfigState, tuple[str, ...]]:
        changed: list[str] = []
        for name in members:
            state, ids = TECHNIQUES[name](state, rng)
            changed.extend(ids)
        # de-duplicate while preserving order
        seen: dict[str, None] = {}
        for cid in changed:
            seen.setdefault(cid, None)
        return state, tuple(seen.keys())

    return combined


TECHNIQUES: dict[str, Technique] = {
    "static_baseline": static_baseline,
    "network_shuffling": network_shuffling,
    "port_rotation": port_rotation,
    "endpoint_mutation": endpoint_mutation,
    "deceptive_presentation": deceptive_presentation,
    "combined_mtd_deception": make_combined(),
}


def get_technique(condition: str) -> Technique:
    """Look up the technique transform for a condition name."""

    try:
        return TECHNIQUES[condition]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown MTD condition: {condition!r}") from exc
