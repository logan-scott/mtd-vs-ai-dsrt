"""Descriptive summary of the planner's recorded action trace.

**DESCRIPTIVE ONLY -- and structurally so.** Nothing here produces a p-value, an
effect size or a confidence interval, so there is nothing for Benjamini-Hochberg to
correct. It cannot enter the FDR family for three independent, checkable reasons:
``analysis.pipeline._fdr_secondary`` takes only ``(rq1r, rq2r)`` and reads p-values
out of those two dicts alone; family membership -- and therefore the BH denominator
-- is the module-level ``SECONDARY_FAMILIES`` literal, fixed before any data is
seen; and this summary is attached as a sibling key of the outer results dict, which
``_fdr_secondary`` never receives. ``tests/test_analysis.py`` pins all three.

The trace columns are OPTIONAL. Every dataset written before action-trace logging
existed lacks them entirely (``analysis.dataset.load_csv`` cannot invent a column
that is not in the file), and the same dataset read through
``measurement.run_sheet.read_csv`` carries the explicit ``NOT_RECORDED`` sentinel
instead. Detection is therefore VALUE-based, never column-presence-based: an
all-sentinel frame has all five columns present and still carries no trace. Both
cases report "not recorded" rather than a fabricated distribution over zero
actions -- a trial in which the planner genuinely chose nothing is a real
observation, and the report must never conflate the two.
"""

from __future__ import annotations

import math

import pandas as pd

from analysis.dataset import TRACE_ANCHOR
from measurement.action_trace import trace_summary
from measurement.models import NOT_RECORDED

#: The five instrumentation columns (``measurement.models.TrialRow`` tail order).
TRACE_COLUMNS: tuple[str, ...] = (
    "action_sequence",
    "n_actions",
    "n_recon_actions",
    "n_exploit_actions",
    "planner_terminal_reason",
)

#: Counts stored directly on the row.
_COUNT_COLUMNS: tuple[str, ...] = ("n_actions", "n_recon_actions", "n_exploit_actions")

#: Counts recovered by decoding ``action_sequence``. ``n_client_coerced_stops`` is the one
#: that keeps ``planner_stop`` honest: a stop the CLIENT substituted for a malformed
#: model response is not a planner that decided to give up, and the terminal reason
#: alone cannot tell them apart. ``n_client_coerced`` counts every action the client
#: substituted, including (prompt format 2) the scan it runs when the model asks to exploit
#: or stop before its attempt's first scan.
_DECODED_COLUMNS: tuple[str, ...] = (
    "n_stop_actions",
    "n_unknown_actions",
    "n_not_executed",
    "n_client_coerced",
    "n_client_coerced_stops",
)

_EMPTY_CELL_NOTE = "no trial in this cell carries a recorded action trace"

_NOT_RECORDED_NOTE = (
    "This dataset carries no planner action trace: the trace columns are absent from the "
    "CSV, or every row holds the explicit '" + NOT_RECORDED + "' sentinel. That is what a "
    "dataset collected before action-trace logging looks like. No distribution and no mean "
    "is reported, because a fabricated zero would be indistinguishable from a trial in "
    "which the planner genuinely took no actions."
)

_READING_NOTE = (
    "Descriptive only: the distribution of the planner's loop-terminal reason and the mean "
    "action counts, per condition and cross-tabbed with the co-primary outcome "
    "exploitation_success. It enters no hypothesis test and no FDR family. To ask why a "
    "condition's failures failed, read the (condition, exploitation_success=0) cell of "
    "by_condition_x_exploitation_success: terminal_reason_counts separates a planner that "
    "stopped ('planner_stop') from one that ran out of budget "
    "('action_budget_exhausted') or clock ('duration_limit'), "
    "n_zero_exploit_action_trials counts the trials that never even attempted an exploit, "
    "n_client_coerced_stops_mean warns when the 'stops' were substituted by the client for "
    "malformed model responses rather than chosen by the model, and n_client_coerced_mean "
    "counts every action the client substituted (from prompt format 2 also the scan it runs "
    "when the model asks to exploit or stop before its attempt's first scan)."
)


def _label(value: object) -> str | None:
    """Group label as a plain JSON scalar; a missing key stays explicitly null."""

    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - array-valued key
        missing = False
    return None if missing else str(value)


def _num(value: object) -> float | None:
    """Plain finite ``float`` or ``None``.

    ``analysis.pipeline.write_report`` serializes with ``allow_nan=False``, so a numpy
    scalar, a NaN or an infinity leaking out of here aborts a real run at the very end.
    """

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _recorded_mask(df: pd.DataFrame) -> pd.Series:
    """Per row: does this trial carry a real recorded trace?

    Keyed off ``planner_terminal_reason`` alone (see ``analysis.dataset.TRACE_ANCHOR``):
    an attempt loop always ends exactly one way, so every recorded row has a non-empty
    reason, while both not-recorded encodings -- column absent, or column present and
    sentinel-valued -- are covered by the same predicate.
    """

    if TRACE_ANCHOR not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    text = df[TRACE_ANCHOR].where(df[TRACE_ANCHOR].notna(), "").astype(str).str.strip()
    return (text != "") & (text != NOT_RECORDED)


def _decoded_counts(df: pd.DataFrame, recorded: pd.Series) -> pd.DataFrame:
    """Decode ``action_sequence`` into extra per-row counts (NaN where unavailable).

    A RECORDED row whose ``action_sequence`` cell is empty is the genuine zero-decision
    trace, and ``load_csv`` renders that cell as ``NaN`` -- so it is mapped back to ``""``
    before decoding. Skipping it instead would silently drop exactly the trials this
    instrumentation exists to characterize.
    """

    blank = pd.DataFrame(
        {name: pd.Series(float("nan"), index=df.index, dtype=float) for name in _DECODED_COLUMNS}
    )
    if "action_sequence" not in df.columns:
        return blank
    cells = df["action_sequence"].where(df["action_sequence"].notna(), "").astype(str)
    for key in df.index[recorded.to_numpy()]:
        summary = trace_summary(cells.at[key])
        if "note" in summary:  # unparseable or not recorded -> stays NaN, never 0
            continue
        for name in _DECODED_COLUMNS:
            blank.at[key, name] = float(summary[name])
    return blank


def _mean_sd(values: pd.Series) -> tuple[float | None, float | None]:
    """Mean and sample SD over the non-missing values, or ``(None, None)``."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None, None
    return _num(numeric.mean()), (_num(numeric.std(ddof=1)) if len(numeric) > 1 else None)


def _cell(g: pd.DataFrame, recorded: pd.Series, decoded: pd.DataFrame) -> dict:
    """Descriptives for one cell, computed over its RECORDED rows ONLY."""

    rec = g.loc[recorded.reindex(g.index, fill_value=False)]
    block: dict = {
        "n": int(len(g)),
        "n_recorded": int(len(rec)),
        "n_not_recorded": int(len(g) - len(rec)),
    }
    if rec.empty:
        # NULL, never 0.0: "not measured" and "the planner never exploited" are
        # different claims and the report must not conflate them.
        block["terminal_reason_counts"] = {}
        block["terminal_reason_props"] = {}
        for col in (*_COUNT_COLUMNS, *_DECODED_COLUMNS):
            block[f"{col}_mean"] = None
            block[f"{col}_sd"] = None
        block["n_zero_exploit_action_trials"] = None
        block["prop_zero_exploit_action_trials"] = None
        block["note"] = _EMPTY_CELL_NOTE
        return block

    reasons = rec[TRACE_ANCHOR].astype(str).str.strip()
    counts = reasons.value_counts()
    block["terminal_reason_counts"] = {str(k): int(v) for k, v in sorted(counts.items())}
    block["terminal_reason_props"] = {str(k): int(v) / len(rec) for k, v in sorted(counts.items())}
    for col in _COUNT_COLUMNS:
        source = rec[col] if col in rec.columns else pd.Series(dtype=float)
        block[f"{col}_mean"], block[f"{col}_sd"] = _mean_sd(source)
    decoded_rec = decoded.reindex(rec.index)
    for col in _DECODED_COLUMNS:
        block[f"{col}_mean"], block[f"{col}_sd"] = _mean_sd(decoded_rec[col])
    exploits = (
        pd.to_numeric(rec["n_exploit_actions"], errors="coerce").dropna()
        if "n_exploit_actions" in rec.columns
        else pd.Series(dtype=float)
    )
    block["n_zero_exploit_action_trials"] = int((exploits == 0).sum()) if len(exploits) else None
    block["prop_zero_exploit_action_trials"] = (
        _num((exploits == 0).mean()) if len(exploits) else None
    )
    return block


def summarize_action_trace(df: pd.DataFrame) -> dict:
    """Per-condition planner-trace descriptives, plus the failing-trial cross-tab.

    Degrades to a ``{"note": ...}`` form rather than raising on every degenerate input
    the pipeline can actually be handed: no ``condition`` column, an empty frame, a
    legacy frame with the trace columns absent, an all-sentinel frame, duplicate index
    labels (a resumed run concatenated back together) and NaN group keys.
    """

    if "condition" not in df.columns:
        return {"note": "no 'condition' column; the action-trace summary needs one"}

    present = [c for c in TRACE_COLUMNS if c in df.columns]
    missing = [c for c in TRACE_COLUMNS if c not in df.columns]

    # A caller-supplied frame may carry duplicate index labels (e.g. a resumed run
    # concatenated back together). ``_cell`` reindexes the recorded mask onto each
    # group's index, which RAISES on duplicates -- so normalize the index once, here,
    # rather than letting a real run die inside a descriptive block.
    df = df.reset_index(drop=True)

    n_trials = int(len(df))
    recorded = _recorded_mask(df)
    n_recorded = int(recorded.sum())

    out: dict = {
        "descriptive_only": True,
        "recorded": bool(n_recorded),
        "coverage": (
            "full"
            if n_trials and n_recorded == n_trials
            else ("none" if not n_recorded else "partial")
        ),
        "n_trials": n_trials,
        "n_recorded": n_recorded,
        "n_not_recorded": n_trials - n_recorded,
        "columns_present": present,
        "columns_missing": missing,
    }
    if not n_recorded:
        out["terminal_reasons_observed"] = []
        out["terminal_reason_counts"] = {}
        out["by_condition"] = []
        out["by_condition_x_exploitation_success"] = []
        out["note"] = _NOT_RECORDED_NOTE
        return out

    reasons = df.loc[recorded, TRACE_ANCHOR].astype(str).str.strip()
    # Derived from the data, not from a hard-coded enum: the terminal-reason vocabulary
    # is owned by the attacker layer, and hard-coding it here would make the analysis
    # silently drop a reason the planner later adds.
    observed = reasons.value_counts()
    out["terminal_reasons_observed"] = sorted(str(r) for r in observed.index)
    out["terminal_reason_counts"] = {str(k): int(v) for k, v in sorted(observed.items())}

    decoded = _decoded_counts(df, recorded)
    # ``dropna=False`` in BOTH blocks: with the default the marginal would silently drop
    # NaN-condition rows while the cross-tab kept them, and the two would stop summing to
    # ``n_trials``.
    out["by_condition"] = [
        {"condition": _label(cond), **_cell(g, recorded, decoded)}
        for cond, g in df.groupby("condition", sort=True, dropna=False)
    ]
    cross: list[dict] = []
    if "exploitation_success" in df.columns:
        keyed = df.assign(_es=pd.to_numeric(df["exploitation_success"], errors="coerce"))
        for (cond, es), g in keyed.groupby(["condition", "_es"], sort=True, dropna=False):
            cross.append(
                {
                    "condition": _label(cond),
                    "exploitation_success": (None if pd.isna(es) else int(es)),
                    **_cell(g, recorded, decoded),
                }
            )
    out["by_condition_x_exploitation_success"] = cross
    out["note"] = _READING_NOTE
    return out
