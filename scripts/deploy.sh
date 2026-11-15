#!/usr/bin/env bash
# One deploy entry point for the isolated testbed (run on the control host).
#
# Sequences the whole provisioning flow and re-seals the testbed before returning:
#   1. build the deception-responder image OFFLINE (control host has egress)
#   2. assemble the attacker-node offline bundle (instrument source + wheelhouse)
#   3. run `cli.py deploy` which: (egress mode) opens temporary provisioning egress,
#      runs site.yml, then ALWAYS closes egress and runs the isolation preflight —
#      failing closed if the testbed is not sealed.
#
# The VMs + dual-homed control host must already exist (terraform apply is a separate,
# gated step — see infra/README.md). A trial must NOT run until this exits 0 (the
# preflight passed). Idempotent and re-runnable.
#
# Env: CONFIG (experiment config), INVENTORY (ansible inventory), MODE (egress|mirror).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"    # -> the repository root
cd "$ROOT"

CONFIG="${CONFIG:-harness/experiment_config.yaml}"
INVENTORY="${INVENTORY:-infra/ansible/inventory.ini}"
MODE="${MODE:-egress}"
PYTHON="${PYTHON:-python}"

[ -f "$CONFIG" ] || { echo "ERROR: config $CONFIG not found (copy the example + fill live section)"; exit 1; }
[ -f "$INVENTORY" ] || { echo "ERROR: inventory $INVENTORY not found (copy inventory.example.ini)"; exit 1; }

echo "==> [1/4] Building the local target images offline (responder + vsftpd backdoor)"
# Docker is needed on the control host ONLY for these build steps. It forces
# net.ipv4.ip_forward=1 while running, so we start it for the builds and stop it
# before provisioning — otherwise the post-deploy seal can never confirm ip_forward=0.
sudo systemctl start docker
bash "$ROOT/infra/ansible/roles/targets/files/deception-responder/build.sh"
bash "$ROOT/infra/ansible/roles/targets/files/vsftpd-backdoor/build.sh"

echo "==> [2/4] Assembling the attacker-node offline bundle (source + wheels)"
$PYTHON -m harness.offline_bundle --root "$ROOT"

echo "==> [3/4] Stopping Docker so the control host can seal (ip_forward=0)"
sudo systemctl stop docker docker.socket

echo "==> [4/4] Provisioning (mode=$MODE) + re-seal + isolation preflight (fail-closed)"
$PYTHON cli.py deploy --config "$CONFIG" --inventory "$INVENTORY" --mode "$MODE"

echo "==> deploy complete and testbed sealed. A live pilot may now run:"
echo "    $PYTHON cli.py pilot --live --config $CONFIG"
