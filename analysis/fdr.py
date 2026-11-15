"""Benjamini-Hochberg false discovery rate control, applied within each RQ."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def benjamini_hochberg(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Apply BH-FDR to a name->p-value mapping.

    Returns name -> {p, p_adj, reject}. p-values that are None/NaN are passed
    through untouched (excluded from the correction).
    """

    names = [k for k, v in pvalues.items() if v is not None and v == v]  # drop None/NaN
    if not names:
        return {k: {"p": v, "p_adj": None, "reject": None} for k, v in pvalues.items()}

    ps = [pvalues[k] for k in names]
    reject, p_adj, _, _ = multipletests(ps, alpha=alpha, method="fdr_bh")
    out = {
        name: {"p": float(pvalues[name]), "p_adj": float(padj), "reject": bool(rej)}
        for name, padj, rej in zip(names, p_adj, reject, strict=True)
    }
    for k, v in pvalues.items():
        if k not in out:
            out[k] = {"p": v, "p_adj": None, "reject": None}
    return out


def holm(pvalues: dict[str, float | None]) -> dict[str, float]:
    """Holm step-down adjusted p-values (strong FWER control) for a name->p mapping.

    Members whose p is None or not finite were not tested: they are left out of the family
    and out of the result (counting a NaN would inflate every other adjusted p).
    """

    finite = {k: float(v) for k, v in pvalues.items() if v is not None and np.isfinite(v)}
    items = sorted(finite.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * float(p)))
        adj[name] = running
    return adj


def bh_adjust(pvalues: list[float], family_size: int) -> list[float]:
    """BH-adjusted p-values for ``pvalues`` against a family of ``family_size``.

    ``family_size`` (m) may exceed ``len(pvalues)`` (k): that is the case where some
    members of a pre-registered family produced no p-value at all. The adjustment is
    ``p_adj_(i) = min(1, min_{j>=i} m * p_(j) / j)``, i.e. ordinary BH with the
    denominator pinned to the declared family size rather than to however many tests
    happened to be runnable.

    ``multipletests(..., method="fdr_bh")`` is not reused here because it clips at
    1.0 *after* taking the running minimum, so rescaling its output by m/k is exact
    only when nothing was clipped. This uses statsmodels' own arithmetic form
    (``p / ecdf`` with ``ecdf = arange(1, k+1)/k``) so that at ``m == k`` it is
    bit-identical to the library; that identity is pinned by a regression test.
    """

    ps = np.asarray(pvalues, dtype=float)
    k = ps.size
    if k == 0:
        return []
    order = np.argsort(ps, kind="stable")
    ordered = ps[order]
    ecdf = np.arange(1, k + 1) / float(k)
    raw = ordered / ecdf * (float(family_size) / float(k))
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    out = np.empty(k, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return [float(x) for x in out]


def bh_family(
    pvalues: dict[str, float | None],
    *,
    alpha: float = 0.05,
    skipped: dict[str, dict] | None = None,
    family_size: int | None = None,
) -> dict:
    """BH over a **pre-registered** family, with untestable members accounted for.

    ``pvalues`` maps every pre-registered outcome in the family to its p-value or to
    ``None``/NaN when no test was performed. ``skipped`` supplies the reason for the
    members already known to be untestable.

    Two decisions are made here, both deliberately independent of the observed data:

    1. **The denominator is the pre-registered family size** (``family_size``,
       defaulting to ``len(pvalues)``), not the number of tests that turned out to be
       runnable. BH controls E[V/R] over the m hypotheses in the family; shrinking m
       after seeing which outcomes degenerated would make every surviving test more
       significant on data-dependent grounds -- the exact pattern this module exists
       to avoid. Retaining an untestable member is conservative: it contributes
       nothing to R and can never be rejected, so FDR control is preserved at the
       price of slightly larger adjusted p-values.
    2. **A skipped outcome is absent from ``tests`` and named in ``skipped``.** It is
       never emitted with ``reject: null``, which would read as "tested and
       inconclusive" for a hypothesis that was never tested, and it is never silently
       deleted, which would hide a deviation from the analysis plan.

    ``tests`` therefore contains exactly the tests performed, ``skipped`` names every
    member that was not, ``n_planned`` records the pre-registered size, and
    ``bh_denominator`` records the m actually used so a reader can check the
    arithmetic.

    ``benjamini_hochberg`` above is deliberately left alone: it is a pure
    implementation of the procedure, shared with ``rq2._binary`` and ``rq4``, where
    every p-value is a real float. Family *membership* is a reporting judgement, not
    a property of the BH algorithm, so it is resolved here instead.
    """

    skipped_out = {k: dict(v) for k, v in (skipped or {}).items()}
    usable: dict[str, float] = {}
    for name, p in pvalues.items():
        if name in skipped_out:
            continue
        if p is None or p != p:
            skipped_out[name] = {
                "reason": "p_unavailable",
                "detail": "the primary model produced no p-value for this outcome",
            }
        else:
            usable[name] = float(p)

    m = int(family_size) if family_size is not None else len(pvalues)
    m = max(m, len(usable))
    names = list(usable)
    adj = bh_adjust([usable[n] for n in names], m)
    tests = {
        n: {"p": usable[n], "p_adj": a, "reject": bool(a <= alpha)}
        for n, a in zip(names, adj, strict=True)
    }
    return {
        "method": "benjamini_hochberg",
        "alpha": float(alpha),
        "n_planned": len(pvalues),
        "n_tests": len(tests),
        "bh_denominator": m,
        "denominator_basis": (
            "pre-registered family size (untestable outcomes are retained in the "
            "denominator, so the correction never loosens in response to the data)"
        ),
        "tests": tests,
        "skipped": dict(sorted(skipped_out.items())),
    }
