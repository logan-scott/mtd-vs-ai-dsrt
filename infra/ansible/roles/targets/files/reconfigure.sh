#!/usr/bin/env bash
# Idempotent per-attempt reconfiguration of the targets node to an MTD condition.
#
# Called by infra/ansible/reconfigure.yml (which the harness AnsibleProvisioner
# runs once per attempt) with the desired observable state as JSON. It renders a
# docker-compose override (every service's published ports — observable AND auxiliary,
# genuine AND decoy) + the deception responder config (decoys' ports/endpoints) from
# that state and reconciles the running containers. FAST + IDEMPOTENT: if the desired config_checksum matches the
# last applied, it is a no-op. Confined to 172.31.66.0/24.
#
# Usage: reconfigure.sh <desired_state.json>
set -euo pipefail

STATE="${1:?desired_state.json}"
# TESTBED / BIND_HOST are overridable via the environment, defaulting to the production
# staging dir and the targets-node bind IP. Production (the reconfigure.yml playbook) sets
# NEITHER, so its behavior is byte-identical to before; the reconfiguration-cost benchmark
# (harness.reconfig_benchmark) sets them to time the REAL reconfigure.sh against a staging
# copy without hardcoding the node's own paths.
TESTBED="${TESTBED:-/opt/testbed}"
BIND_HOST="${BIND_HOST:-172.31.66.10}"
STATE_DIR="${TESTBED}/state"
mkdir -p "$STATE_DIR"

CHECKSUM="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["config_checksum"])' "$STATE")"
CURRENT_FILE="${STATE_DIR}/current.checksum"
if [ -f "$CURRENT_FILE" ] && [ "$(cat "$CURRENT_FILE")" = "$CHECKSUM" ]; then
  echo "reconfigure: already at ${CHECKSUM} (no-op)"
  exit 0
fi

# Render the compose override (EVERY service -> observable_port:container_port plus any
# fixed auxiliary_ports, decoys included as the single `decoys` compose service) and the
# responder banner map from the desired state. The rendering lives in render_override.py
# (staged by the targets role) so it is unit-tested offline; it publishes each service's
# real in-container port, carries through auxiliary ports the !override merge tag would
# otherwise drop (vsftpd's 6200 backdoor shell), and refuses a duplicate host port across
# genuine services, decoys and auxiliary ports alike.
python3 "${TESTBED}/render_override.py" "$STATE" "$TESTBED" "$BIND_HOST"

# Reconcile containers to the rendered state (idempotent; only changed ones recreate).
docker compose -f "${TESTBED}/docker-compose.yml" -f "${TESTBED}/docker-compose.override.yml" \
  up -d --remove-orphans

# The decoys container bind-mounts responder.yml, so `up -d` does NOT recreate it when only
# the banner map changed — signal the responder to reload it (deceptive_presentation banner/
# endpoint rotation). A rotated decoy PORT *does* change the override's published ports, so
# `up -d` above recreates the container and it starts on the rotated ports; the SIGHUP is the
# banner/endpoint path and is harmless (a no-op reload) on a just-recreated container. The
# harness verifies the rotated decoy ports are actually reachable after this script runs
# (AnsibleProvisioner._verify_decoys_published) and fails the attempt if they are not.
docker compose -f "${TESTBED}/docker-compose.yml" -f "${TESTBED}/docker-compose.override.yml" \
  kill -s SIGHUP decoys 2>/dev/null || true

echo "$CHECKSUM" >"$CURRENT_FILE"
echo "reconfigure: applied ${STATE} (checksum ${CHECKSUM})"
