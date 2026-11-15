"""Per-trial decoy-engagement instrumentation, derived from the exploit deliveries.

**PURE INSTRUMENTATION.** Nothing here produces a p-value, an effect size or a
confidence interval, and none of the three columns it fills enters a hypothesis test
or the Benjamini-Hochberg family (pinned in ``tests/test_analysis.py``).

**Why it exists.** "The attacker engaged a decoy" and "the attacker *took the bait*"
are different claims, and the recorded outcomes alone cannot separate them: in the
rotating arms the attacker aims at a decoy in 36-49% of trials, yet its FIRST exploit
delivery is decoy-aimed in only 0-2.6% of them. The first pattern is post-failure
fallback -- the attacker exhausts what it believes are genuine services and then tries
what is left; the second is deception working as designed. Recording first-choice,
first-decoy ordinal and decoy-aimed count per trial makes the two separable directly,
without re-parsing ``action_sequence``.

**Nothing here is a new measurement.** ``ExploitDelivery.target_genuine`` is stamped
from ground truth when the exploit is fired and ``reached_and_executed`` is measured for real
(under the wall clock, then scored at the delivery's completion: ``harness/live_contract.py``
§2); this module only classifies and counts records that already exist. One consequence of
that scoring: a delivery overtaken by a reconfiguration reads ``genuine`` if it was aimed at a
genuine service (``target_genuine`` is judged at firing) but ``unresolved`` if it was aimed at
a decoy, like any stale tuple. The analysis's name rule (below) classifies both by name.

**The classification rule is the offline/live contract's own truth table**
(``harness/live_contract.py`` §2), which both executors implement identically::

    target in genuine  -> (reached_and_executed=True,  target_genuine=True )  "genuine"
    target in decoys   -> (reached_and_executed=True,  target_genuine=False)  "decoy"
    otherwise (stale)  -> (reached_and_executed=False, target_genuine=False)  "unresolved"

So ``target_genuine`` alone is NOT sufficient: it is False both for a decoy and for a
stale identity that MTD had already moved out from under the attacker. Counting the
stale case as decoy-aimed would inflate exactly the bait-taking rate this module exists
to measure honestly, which is why :data:`TARGET_UNRESOLVED` is a first-class outcome
rather than folded into either neighbour. ``target_genuine`` is checked FIRST, so a
genuine service the attacker aimed at but could not reach still counts as genuine-aimed:
the columns record what the planner AIMED at, never whether it succeeded.

**The analysis classifies by name instead (decision A2, 2026-10-31).** A stale tuple is
``unresolved`` here, and under the real clock most first exploits in the moving conditions are
stale, so these columns say little there. They stay as recorded; the analysis
(``analysis/targeting.py``, ``rq2.decoy_targeting``) classifies every executed exploit again by
its target's SERVICE NAME from ``action_sequence`` (names never change under any technique).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from measurement import clock
from measurement.models import ExploitDelivery

__all__ = [
    "DECOY_ENGAGEMENT_COLUMNS",
    "NO_EXPLOIT_DELIVERY",
    "TARGET_DECOY",
    "TARGET_GENUINE",
    "TARGET_UNRESOLVED",
    "DecoyEngagement",
    "classify_delivery",
    "decoy_engagement",
]

#: The ``TrialRow`` columns this module fills, in canonical (tail) order.
DECOY_ENGAGEMENT_COLUMNS: tuple[str, ...] = (
    "first_exploit_target",
    "first_decoy_delivery_ordinal",
    "n_decoy_targeted_deliveries",
)

# What the first delivery aimed at. Lowercase, like every other recorded vocabulary in
# the dataset, so a real observation can never be confused with a SCREAMING sentinel.
TARGET_GENUINE = "genuine"
TARGET_DECOY = "decoy"
TARGET_UNRESOLVED = "unresolved"

#: Explicit "this trial made NO exploit delivery" sentinel -- a real observation about a
#: trial whose planner never delivered an exploit, and NOT the same fact as
#: ``measurement.models.NOT_RECORDED`` ("this row predates the columns"). SCREAMING_CASE
#: so it cannot collide with the recorded vocabulary above.
NO_EXPLOIT_DELIVERY = "NO_EXPLOIT_DELIVERY"


@dataclass(frozen=True)
class DecoyEngagement:
    """The three derived facts (see :func:`decoy_engagement`).

    ``first_decoy_delivery_ordinal`` is ``None`` when no delivery was decoy-aimed; the
    "this row was never instrumented" case is carried by ``first_exploit_target``, the
    group's anchor, never by this field alone.
    """

    first_exploit_target: str
    first_decoy_delivery_ordinal: int | None
    n_decoy_targeted_deliveries: int


def classify_delivery(delivery: ExploitDelivery) -> str:
    """Classify ONE delivery as :data:`TARGET_GENUINE` / DECOY / UNRESOLVED.

    Implements the contract truth table in the module docstring. ``target_genuine`` is
    authoritative for the genuine case (it is the ground-truth-stamped field); the decoy
    case is the one the table distinguishes by reach, because only a service that was
    actually there answers.
    """

    if delivery.target_genuine:
        return TARGET_GENUINE
    return TARGET_DECOY if delivery.reached_and_executed else TARGET_UNRESOLVED


def decoy_engagement(deliveries: Sequence[ExploitDelivery]) -> DecoyEngagement:
    """Summarize one trial's exploit deliveries into the three persisted facts.

    Deliveries are ordered by timestamp -- the same chronological ordering
    ``security_outcomes.attempts_per_success`` uses, with the same STABLE tie-breaking
    (equal timestamps keep their recorded order) and the same 1-BASED ordinal, so the
    returned ordinal and ``attempts_per_success`` index the same sequence and may be
    compared directly.

    An empty sequence returns :data:`NO_EXPLOIT_DELIVERY` with a null ordinal and a count
    of ``0``: the trial genuinely delivered nothing, which is an observation, not a gap.
    """

    ordered = sorted(deliveries, key=lambda d: clock.parse(d.timestamp))
    if not ordered:
        return DecoyEngagement(NO_EXPLOIT_DELIVERY, None, 0)
    kinds = [classify_delivery(d) for d in ordered]
    first_decoy = next(
        (i for i, kind in enumerate(kinds, start=1) if kind == TARGET_DECOY),
        None,
    )
    return DecoyEngagement(
        first_exploit_target=kinds[0],
        first_decoy_delivery_ordinal=first_decoy,
        n_decoy_targeted_deliveries=sum(1 for kind in kinds if kind == TARGET_DECOY),
    )
