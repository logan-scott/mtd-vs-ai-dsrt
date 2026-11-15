#!/usr/bin/env bash
# Stage the instrument code to the SEALED targets node so `cli.py reconfig-bench` can run
# THERE (the benchmark shells out to the local docker + reconfigure.sh; there is no SSH path,
# so it must run on the node, not the control host).
#
# The node cannot `git clone` (no egress) and has only Python 3 + PyYAML — which is the ONLY
# third-party dependency the reconfig-bench path needs (proved by
# tests/test_cli_import_without_analysis_stack.py; the analysis stack is NOT required because
# cli.py imports it lazily). This copies exactly the packages the benchmark imports and nothing
# else: no .venv, no data/, no docs/results, no __pycache__, no .git.
#
# Usage (run on the CONTROL host, from the repository root):
#   scripts/stage_reconfig_bench.sh [TARGETS_HOST] [SSH_PORT] [SSH_USER] [DEST_DIR]
# Defaults match infra/ansible/inventory.example.ini (targets node admin sshd on :2222):
#   TARGETS_HOST=172.31.66.10  SSH_PORT=2222  SSH_USER=labadmin  DEST_DIR=~/mtd-instrument
set -euo pipefail

TARGETS_HOST="${1:-172.31.66.10}"
SSH_PORT="${2:-2222}"
SSH_USER="${3:-labadmin}"
DEST_DIR="${4:-mtd-instrument}"   # relative -> the SSH user's home on the node

HERE="$(cd "$(dirname "$0")/.." && pwd)"  # the repository root
cd "$HERE"

# The packages the reconfig-bench import chain actually needs, plus the CLI entrypoint.
# (harness -> live_runner/safety/reconfig_benchmark; mtd -> techniques/checksums/models;
#  measurement -> models; testbed -> roster.yaml/profile; attacker -> base/tools.)
INCLUDE=(harness mtd measurement testbed attacker cli.py)

echo "staging ${INCLUDE[*]} -> ${SSH_USER}@${TARGETS_HOST}:${DEST_DIR}/ (ssh port ${SSH_PORT})"
rsync -az --delete-excluded \
  -e "ssh -p ${SSH_PORT}" \
  --exclude='.venv' --exclude='.venv*' \
  --exclude='data' --exclude='docs' \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.git' \
  --relative \
  "${INCLUDE[@]}" \
  "${SSH_USER}@${TARGETS_HOST}:${DEST_DIR}/"

echo
# Check the node's interpreter NOW, at staging — the collection path needs Python >= 3.10.
# The bug this check exists to pre-empt surfaced as an ImportError twelve steps into the
# operator procedure; surface it here instead, with the version actually found.
echo "checking the node's Python (collection floor is 3.10) ..."
NODE_PY="$(ssh -p "${SSH_PORT}" "${SSH_USER}@${TARGETS_HOST}" \
  'python3 -c "import sys;print(\"%d.%d.%d\"%sys.version_info[:3])"' 2>/dev/null || true)"
if [ -z "${NODE_PY}" ]; then
  echo "ERROR: could not determine the node's Python 3 version (is python3 on PATH on ${TARGETS_HOST}?)." >&2
  exit 1
fi
if ! ssh -p "${SSH_PORT}" "${SSH_USER}@${TARGETS_HOST}" \
  'python3 -c "import sys;sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)"'; then
  echo "ERROR: node Python is ${NODE_PY}; reconfig-bench requires >= 3.10 (see pyproject.toml" >&2
  echo "       requires-python and docs/reconfig_benchmark_runbook.md §0). Upgrade the node's" >&2
  echo "       python3 before running the benchmark." >&2
  exit 1
fi
echo "node Python ${NODE_PY} OK (>= 3.10)."
echo

# Record WHICH code was staged. The node gets no .git, so without this the benchmark's
# metadata.json says git_commit: null (C235). harness/reconfig_benchmark.py reads this marker
# (git_commit_source: staged_marker) whenever it is not running inside a git checkout.
MARKER="$(python3 - <<'PY'
import datetime, json, subprocess

def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None

head = (git("rev-parse", "HEAD") or "").strip() or None
status = git("status", "--porcelain", "--", ".")
print(json.dumps({
    "git_commit": head,
    "git_uncommitted": None if status is None else [l.rstrip() for l in status.splitlines() if l.strip()],
    "staged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}))
PY
)"
printf '%s\n' "${MARKER}" | ssh -p "${SSH_PORT}" "${SSH_USER}@${TARGETS_HOST}" \
  "cat > ${DEST_DIR}/STAGED_COMMIT.json"
echo "recorded the staged code version on the node (${DEST_DIR}/STAGED_COMMIT.json):"
echo "  ${MARKER}"
echo
echo "staged. Now, ON THE NODE:"
echo "  ssh -p ${SSH_PORT} ${SSH_USER}@${TARGETS_HOST}"
echo "  cd ${DEST_DIR}"
echo "  python3 cli.py reconfig-bench --testbed-dir /opt/testbed \\"
echo "      --targets-host ${TARGETS_HOST} --testbed-cidr 172.31.66.0/24 \\"
echo "      --replicates 20 --dry-run      # confirm it resolves, then drop --dry-run"
