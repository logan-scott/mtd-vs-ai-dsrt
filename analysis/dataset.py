"""Load and prepare the trial-level dataset for analysis.

Consolidates run-sheet rows (or a written CSV) into a single DataFrame keyed by
condition and attempt index, screens a-priori-excluded trials, and adds the
derived columns the analyses use (``defense`` = static vs. mtd; ``mtd`` group).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from measurement.models import NOT_RECORDED, TrialRow
from measurement.run_sheet import TRIAL_COLUMNS, row_to_dict

STATIC = "static_baseline"

# Anchor column for "does this dataset carry the planner action trace at all".
# ``planner_terminal_reason`` and NOT ``action_sequence``: pandas types a column of empty
# strings as all-NaN float64, so a legitimately recorded run in which every trial produced
# an empty sequence would be indistinguishable from a pre-instrumentation dataset on
# ``action_sequence``. Every RECORDED row carries a non-empty terminal reason (the attempt
# loop always ends exactly one way), so this column is never silently empty.
TRACE_ANCHOR = "planner_terminal_reason"


def action_trace_recorded(df: pd.DataFrame) -> bool:
    """True iff this frame comes from a run that recorded the planner's action sequence.

    Three cases, all of which occur in practice:

    * the column is ABSENT -- an archived CSV written before the instrumentation existed,
      loaded through :func:`load_csv` (pandas cannot invent a column that is not in the
      file);
    * the column is present and every value is ``NOT_RECORDED`` -- the same legacy dataset
      after :func:`measurement.run_sheet.read_csv` substituted the sentinel;
    * the column is present with real terminal reasons -- the trace was recorded.

    Returns ``False`` for the first two, so callers emit a ``{"note": ...}`` rather than a
    fabricated all-zero trace summary.
    """

    if TRACE_ANCHOR not in df.columns or len(df) == 0:
        return False
    values = df[TRACE_ANCHOR].astype("string")
    return not bool((values.isna() | (values == NOT_RECORDED)).all())


# ``service_availability_impact`` (frozen dataset column) stores availability RETAINED --
# the proportion of legitimate requests answered validly within the timeout -- so HIGHER IS
# BETTER, despite a name that reads like a degradation measure (metrics_specification 3.4).
# The dataset column is authoritative and is NOT renamed; the analysis layer derives an
# explicitly cost-oriented twin for every cost/overhead-oriented use instead.
AVAILABILITY_RETAINED = "service_availability_impact"
AVAILABILITY_LOSS = "service_availability_loss"


def availability_loss(df: pd.DataFrame) -> pd.Series:
    """Cost-oriented availability: ``1 - service_availability_impact`` (higher = worse)."""

    if AVAILABILITY_RETAINED not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return 1.0 - pd.to_numeric(df[AVAILABILITY_RETAINED], errors="coerce")


def to_dataframe(rows: list[TrialRow]) -> pd.DataFrame:
    """Build a trial-level DataFrame from run-sheet rows (canonical columns)."""

    df = pd.DataFrame([row_to_dict(r) for r in rows], columns=TRIAL_COLUMNS)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a written trial dataset CSV (empty cells -> NaN).

    ``float_precision="round_trip"`` so parsed floats are bit-identical to what
    ``run_sheet.write_csv`` serialized. Pandas' default fast parser can differ in the last
    ULP (e.g. -0.12188222103697299 reads back as -0.1218822210369729), which would make a
    CSV-driven re-analysis silently non-reproducible against the in-memory run it came from.
    """

    return pd.read_csv(path, float_precision="round_trip")


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only included trials (a-priori exclusion screening)."""

    return df.loc[df["excluded"] == 0].copy()


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add analysis grouping columns."""

    out = df.copy()
    out["defense"] = out["condition"].apply(lambda c: "static" if c == STATIC else "mtd")
    # Guarded: an unconditional assignment would create an all-NaN column when the source
    # is absent, which silently destroys every complete-case analysis that lists it.
    if AVAILABILITY_RETAINED in out.columns:
        out[AVAILABILITY_LOSS] = availability_loss(out)
    return out


def to_frame(rows_or_df) -> pd.DataFrame:
    """Normalize ``list[TrialRow] | DataFrame`` to a RAW (UNSCREENED) DataFrame.

    The single normalizer shared by ``prepare`` (which then screens) and by
    ``analysis.exclusions`` / ``analysis.pipeline`` (which need the pre-screen frame to
    count what the screen removed). ``screen()`` and ``add_derived()`` both ``.copy()``,
    so handing the caller's own frame straight back never mutates it.
    """

    if isinstance(rows_or_df, pd.DataFrame):
        return rows_or_df
    return to_dataframe(list(rows_or_df))


def prepare(rows_or_df, *, screen_excluded: bool = True) -> pd.DataFrame:
    """One-call preparation: to DataFrame -> screen -> derive columns."""

    df = to_frame(rows_or_df)
    if screen_excluded and "excluded" in df.columns:
        df = screen(df)
    return add_derived(df)


def mtd_conditions(df: pd.DataFrame) -> list[str]:
    """The MTD technique categories present (excludes the static baseline)."""

    return [c for c in df["condition"].unique().tolist() if c != STATIC]
