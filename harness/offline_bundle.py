"""Assemble the offline install bundle the attacker node needs (no-egress install).

``roles/attacker`` installs the instrument with ``pip --no-index --find-links
/opt/wheels`` from a staged source tree — so the no-egress node needs (a) the
instrument **source** (the pip-installable package tree, including
``testbed/roster.yaml`` which the courier reads) and (b) a **wheelhouse** of its
dependencies (the ``[attacker]`` extra + transitive deps). Both are built on the
control host (which has egress) by ``make offline-bundle`` / ``python -m
harness.offline_bundle`` and staged into ``infra/ansible/roles/attacker/files/``.

Only a minimal, deterministic subset of the repo is staged (the package dirs +
``pyproject.toml`` + ``README.md``) — never ``infra/``, ``data/``, ``tests/``, or the
virtualenv — so the assembly is reproducible and cannot recurse into its own output.
The source-staging is pure Python (unit-tested); only the wheel build shells out to
``pip`` (needs egress, run at deploy time).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# The pip-installable package tree (mirrors pyproject packages.find) + top-level
# files pip needs. Deliberately excludes infra/docs/tests/data/.venv.
PACKAGE_DIRS: tuple[str, ...] = ("measurement", "mtd", "attacker", "harness", "analysis", "testbed")
PACKAGE_FILES: tuple[str, ...] = ("pyproject.toml", "README.md")
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "*.egg-info", ".pytest_cache")

# Where the staged bundle lands (what the attacker role's `src:` paths expect).
_ATTACKER_FILES = Path("infra/ansible/roles/attacker/files")

# The attacker node's Python (installed via deadsnakes). The wheelhouse MUST be built
# on this same version: cp3XX dependency wheels (e.g. PyYAML) silently break the
# offline install if the node Python differs. Keep these two equal.
TARGET_PYTHON: tuple[int, int] = (3, 11)
_PROJECT_WHEEL_PREFIX = "mtd_instrument-"


class BundleError(RuntimeError):
    """The offline bundle could not be assembled correctly (fail loudly, not silently)."""


def check_build_python(
    target: tuple[int, int] = TARGET_PYTHON,
    current: tuple[int, int] | None = None,
    platform: str | None = None,
) -> str:
    """Assert the build Python + OS match the node's; return the wheel tag (e.g. cp311).

    A wrong version OR a non-Linux build would stage dependency wheels the Linux node
    cannot install (cp3XX ABI, or macOS/Windows platform wheels) — so this raises rather
    than producing a subtly-broken bundle. Build on the Linux control host's 3.11.
    """

    current = current or sys.version_info[:2]
    if current != target:
        raise BundleError(
            f"offline bundle must be built with Python {target[0]}.{target[1]} to match "
            f"the attacker node, but this interpreter is {current[0]}.{current[1]}; "
            f"run `make offline-bundle` on the control host's Python {target[0]}.{target[1]}"
        )
    plat = platform if platform is not None else sys.platform
    if not plat.startswith("linux"):
        raise BundleError(
            f"offline bundle must be built on Linux (the attacker node's OS) so the "
            f"dependency wheels are manylinux, but this is {plat!r}; build it on the "
            f"control host, not a dev machine"
        )
    return f"cp{target[0]}{target[1]}"


def bundle_paths(instrument_root: str | Path) -> dict[str, Path]:
    """The staged locations the attacker role copies from."""

    base = Path(instrument_root) / _ATTACKER_FILES
    return {"instrument_dir": base / "instrument", "wheels_dir": base / "wheels"}


def stage_instrument_source(instrument_root: str | Path, dest: str | Path) -> list[str]:
    """Copy the minimal installable source tree to ``dest``; return the staged files.

    Deterministic and self-contained: it copies only ``PACKAGE_DIRS`` + ``PACKAGE_FILES``,
    so ``testbed/roster.yaml`` and ``attacker/prompts/system.md`` ship, and nothing
    under ``infra/`` (including this bundle's own output) is ever copied.
    """

    root, dest = Path(instrument_root), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in PACKAGE_FILES:
        shutil.copy2(root / name, dest / name)
    for name in PACKAGE_DIRS:
        shutil.copytree(root / name, dest / name, ignore=_IGNORE)
    return sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())


def pip_wheel_command(
    instrument_root: str | Path, wheels_dir: str | Path, extra: str = "attacker"
) -> list[str]:
    """The ``pip wheel`` command that fills the wheelhouse (instrument + extra deps)."""

    return [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        f"{Path(instrument_root)}[{extra}]",
        "-w",
        str(wheels_dir),
    ]


def project_wheel_command(instrument_root: str | Path, wheels_dir: str | Path) -> list[str]:
    """Build only the ``mtd-instrument`` project wheel (no deps) into the wheelhouse.

    The attacker role installs ``mtd-instrument`` BY NAME from the wheelhouse, so the
    project wheel must be present even if the deps build was cached/partial.
    """

    return [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(Path(instrument_root)),
        "--no-deps",
        "-w",
        str(wheels_dir),
    ]


def build_bundle(instrument_root: str | Path = ".", *, build_wheels: bool = True) -> dict:
    """Assemble the full offline bundle; return a summary of what was staged.

    Fails loudly (``BundleError``) if the build Python differs from the node Python or
    if the ``mtd-instrument`` wheel is not produced (so the attacker installs by name).
    """

    paths = bundle_paths(instrument_root)
    staged = stage_instrument_source(instrument_root, paths["instrument_dir"])

    wheels: list[str] = []
    if build_wheels:
        check_build_python()  # refuse to stage cp3XX wheels the node cannot install
        paths["wheels_dir"].mkdir(parents=True, exist_ok=True)
        # 1. deps + project wheel; 2. the project wheel explicitly (belt-and-suspenders).
        subprocess.run(pip_wheel_command(instrument_root, paths["wheels_dir"]), check=True)
        subprocess.run(project_wheel_command(instrument_root, paths["wheels_dir"]), check=True)
        wheels = sorted(p.name for p in paths["wheels_dir"].glob("*.whl"))
        if not any(w.startswith(_PROJECT_WHEEL_PREFIX) for w in wheels):
            raise BundleError(
                f"the {_PROJECT_WHEEL_PREFIX}*.whl was not built into {paths['wheels_dir']}; "
                "the attacker node installs mtd-instrument by name and would fail"
            )

    return {
        "instrument_dir": str(paths["instrument_dir"]),
        "wheels_dir": str(paths["wheels_dir"]),
        "staged_files": len(staged),
        "roster_included": "testbed/roster.yaml" in staged,
        "wheels": wheels,
        "project_wheel": next((w for w in wheels if w.startswith(_PROJECT_WHEEL_PREFIX)), None),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - deploy entry
    import argparse

    parser = argparse.ArgumentParser(description="Assemble the attacker-node offline bundle")
    parser.add_argument("--root", default=".", help="instrument repo root")
    parser.add_argument(
        "--no-wheels", action="store_true", help="stage source only (skip the pip wheel build)"
    )
    args = parser.parse_args(argv)

    summary = build_bundle(args.root, build_wheels=not args.no_wheels)
    print(
        f"staged {summary['staged_files']} source files "
        f"(roster included: {summary['roster_included']}), "
        f"{len(summary['wheels'])} wheels -> {summary['wheels_dir']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - deploy entry
    raise SystemExit(main())
