"""Run-sheet writer: serialize TrialRow objects to the trial-level dataset.

The canonical column order is the field order of ``TrialRow`` (which mirrors
``docs/metrics_specification.md`` §1–§4). Writers are deterministic: identical
rows produce byte-identical output. Supports CSV (analysis-friendly) and JSONL
(lossless, preserves nulls), plus a JSON schema dump for the data dictionary.

Readers are backward compatible with datasets written before a column existed: see
``OPTIONAL_ON_READ`` and the :func:`read_csv` docstring for the exact contract.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path

from measurement.models import ABSENT_ON_READ, TrialRow

# Canonical column order = TrialRow field declaration order.
TRIAL_COLUMNS: list[str] = [f.name for f in fields(TrialRow)]

# Columns a dataset CSV may legally OMIT, mapped to the value substituted when it does.
# DERIVED from the per-field ``ABSENT_ON_READ`` marker in ``measurement.models`` -- never a
# hand-maintained literal list -- so the strict header check below stays exactly as strict
# as it was for every column that does not carry the marker, and so the substituted value
# lives next to the field's type and default. A misspelled marker key leaves the column
# REQUIRED (fails closed, loudly); nothing can widen this set except an explicit, reviewed
# metadata entry on a ``TrialRow`` field. Marker values must be IMMUTABLE: the same object
# is handed to every row that omits the column.
OPTIONAL_ON_READ: dict[str, object] = {
    f.name: f.metadata[ABSENT_ON_READ] for f in fields(TrialRow) if ABSENT_ON_READ in f.metadata
}

# Columns whose absence still makes a CSV "not a run sheet".
REQUIRED_COLUMNS: list[str] = [c for c in TRIAL_COLUMNS if c not in OPTIONAL_ON_READ]


_FIELD_TYPES: dict[str, str] = {f.name: str(f.type) for f in fields(TrialRow)}
_TRUEISH = frozenset({"1", "true", "yes", "t", "y"})
_FALSEISH = frozenset({"0", "false", "no", "f", "n"})


def _parse_value(column: str, raw: str | None):
    """Coerce one CSV cell back to the column's declared ``TrialRow`` type.

    Mirrors :func:`_csv_value`, which renders ``None`` as ``""``: an empty cell becomes
    ``None`` for an optional field and the field's zero/empty value otherwise. Integer
    columns are parsed EXACTLY (``int(raw)`` first) so a value above 2**53 -- e.g. a
    64-bit ``randomization_seed`` -- survives the round trip; ``int(float(raw))`` alone
    would silently return 9007199254740992 for 9007199254740993.
    """

    declared = _FIELD_TYPES[column]
    if raw is None or raw == "":
        if "None" in declared:
            return None
        return "" if declared.startswith("str") else 0
    if declared.startswith("str"):
        return raw
    text = raw.strip()
    if declared.startswith("int"):
        try:
            return int(text)
        except ValueError:
            pass
        lowered = text.lower()
        if lowered in _TRUEISH:  # ``excluded`` is semantically boolean (TrialMetadata.excluded)
            return 1
        if lowered in _FALSEISH:
            return 0
        return int(float(text))  # tolerate "3.0" written by a third-party tool
    if declared.startswith("float"):
        return float(text)
    return raw


def read_csv(path: str | Path) -> list[TrialRow]:
    """Read a written trial dataset CSV back into ``TrialRow`` objects.

    Lossless in the direction that matters: every value the CSV records survives the read
    unchanged, and nothing absent from it is invented. For a CSV written by the CURRENT
    schema this is byte-level -- ``write_csv(read_csv(p), q)`` reproduces ``p``
    byte-for-byte, so an archived dataset can be re-analyzed without re-running a trial.

    For a LEGACY CSV -- one written before a column was added, e.g. the archived datasets
    under ``docs/results/`` -- the read succeeds (see :data:`OPTIONAL_ON_READ`) and writing
    the rows back UPGRADES the file to the current header, so ``q`` is deliberately NOT
    byte-identical to ``p``: it is ``p`` WIDENED, its own columns unchanged and in order,
    with the absent ones appended and filled by their declared not-recorded value. That
    transform invents no measurement and loses none, and it is idempotent -- ``q`` is
    canonical, so re-reading and re-writing ``q`` IS byte-stable. Byte-identity is
    therefore the special case that holds exactly when ``p`` is already canonical.

    An absent optional column takes its substitute straight from the field marker, never
    from :func:`_parse_value`, which would render a missing cell as ``""``/``0`` -- values
    a genuinely recorded row can also produce.

    Raises ``ValueError`` when a REQUIRED column is missing, so pointing a tool at the
    wrong CSV still fails loudly instead of yielding silently malformed rows.
    """

    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        present = set(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in present]
        if missing:
            raise ValueError(f"{path}: not a run-sheet CSV; missing columns {missing}")
        # Computed ONCE from the header: "column absent from the file" is a property of
        # the file, not of a cell, so a present-but-empty cell keeps its own meaning.
        absent = {c: v for c, v in OPTIONAL_ON_READ.items() if c not in present}
        return [
            TrialRow(
                **{
                    c: (absent[c] if c in absent else _parse_value(c, rec.get(c)))
                    for c in TRIAL_COLUMNS
                }
            )
            for rec in reader
        ]


def row_to_dict(row: TrialRow) -> dict:
    """Ordered dict of a row (canonical column order)."""

    d = asdict(row)
    return {col: d[col] for col in TRIAL_COLUMNS}


def _csv_value(v: object) -> str:
    """Render a cell: None -> empty string; everything else via str()."""

    return "" if v is None else str(v)


def write_csv(rows: list[TrialRow], path: str | Path) -> Path:
    """Write rows to CSV with the canonical header. Returns the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(TRIAL_COLUMNS)
        for row in rows:
            d = row_to_dict(row)
            writer.writerow([_csv_value(d[col]) for col in TRIAL_COLUMNS])
    return path


def write_jsonl(rows: list[TrialRow], path: str | Path) -> Path:
    """Write rows to JSONL (one JSON object per line, nulls preserved)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row_to_dict(row), sort_keys=False) + "\n")
    return path


def _py_type_name(t: object) -> str:
    return getattr(t, "__name__", str(t))


def schema() -> dict:
    """Machine-readable schema: column -> declared type (for the data dictionary)."""

    return {"columns": [{"name": f.name, "type": _py_type_name(f.type)} for f in fields(TrialRow)]}


def write_schema(path: str | Path) -> Path:
    """Dump the schema to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema(), indent=2) + "\n", encoding="utf-8")
    return path
