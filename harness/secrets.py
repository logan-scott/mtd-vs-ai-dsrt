"""Minimal stdlib loader for a gitignored ``.secrets`` file.

Reads ``KEY=VALUE`` lines (dotenv-style) and populates ``os.environ`` so the rest
of the tool can keep reading ``os.environ.get("ANTHROPIC_API_KEY")`` without the
operator having to export anything in the shell. No third-party dependency.

Precedence: a variable already set in the real environment WINS over the file, so
an explicit ``export`` (or CI secret) is never silently overridden. Blank lines,
``#`` comments, optional surrounding quotes, and a leading ``export`` are handled.
The file is optional — a missing file is a no-op.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATH = ".secrets"


def load_secrets(path: str | os.PathLike = DEFAULT_PATH, *, override: bool = False) -> list[str]:
    """Load ``KEY=VALUE`` pairs from ``path`` into ``os.environ``.

    Returns the list of keys applied. Existing environment variables are kept
    unless ``override=True``. A missing file returns an empty list.
    """

    p = Path(path)
    if not p.is_file():
        return []
    applied: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied
