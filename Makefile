# Reproducible commands for the MTD instrument.
# Activate the venv first (python -m venv .venv && source .venv/bin/activate),
# or set PYTHON to a suitable interpreter.

PYTHON ?= python
CONFIG ?= harness/experiment_config.example.yaml

.PHONY: help install test lint fmt pilot run analyze cost-probe stage-reconfig-bench clean dry-run preflight offline-bundle bootstrap-control deploy teardown

# reconfig-bench staging defaults (match infra/ansible/inventory.example.ini).
TARGETS_HOST ?= 172.31.66.10
SSH_PORT ?= 2222
SSH_USER ?= labadmin
DEST_DIR ?= mtd-instrument

help:
	@echo "install          - editable install with dev + analysis extras"
	@echo "test             - run the test suite"
	@echo "lint             - ruff + black --check"
	@echo "fmt              - apply black formatting"
	@echo "pilot            - OFFLINE: generate pilot dataset + analysis report (data/pilot/)"
	@echo "run              - OFFLINE: generate full dataset + analysis report (data/full/)"
	@echo "analyze          - run analysis on an existing dataset CSV (DATASET=...)"
	@echo "cost-probe       - REAL Haiku calls vs offline mock: cost/latency/capability (needs ANTHROPIC_API_KEY, ~few \$$)"
	@echo "stage-reconfig-bench - rsync the instrument to the targets node for reconfig-bench (TARGETS_HOST=...)"
	@echo "clean            - remove generated data/ outputs"
	@echo "dry-run          - exercise the LIVE path with the mock executor; print plan + egress map (no network)"
	@echo "preflight        - assert the live-run isolation invariants on paper (add live probes without --dry-run)"
	@echo "offline-bundle   - stage the attacker-node install bundle (instrument source + wheelhouse)"
	@echo "bootstrap-control- install control-host prereqs (ansible + docker) — run once before deploy"
	@echo "deploy           - provision the (already-provisioned) testbed: build+stage, run site.yml, re-seal, preflight"
	@echo "teardown         - print the Proxmox teardown command (terraform destroy)"

install:
	$(PYTHON) -m pip install -e '.[dev,analysis]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

fmt:
	$(PYTHON) -m black .

pilot:
	$(PYTHON) cli.py pilot --config $(CONFIG)

run:
	$(PYTHON) cli.py run --config $(CONFIG)

analyze:
	$(PYTHON) cli.py analyze --dataset $(DATASET) $(ARGS)

cost-probe:
	$(PYTHON) cli.py cost-probe --config $(CONFIG)

# Stage the instrument code to the sealed targets node so `reconfig-bench` can run THERE
# (it shells out to the node's local docker + reconfigure.sh; there is no SSH path). Copies
# only the packages the benchmark imports; the node needs only Python 3 + PyYAML. See
# docs/reconfig_benchmark_runbook.md for the full flow (stage -> dry-run -> run -> outputs).
stage-reconfig-bench:
	scripts/stage_reconfig_bench.sh $(TARGETS_HOST) $(SSH_PORT) $(SSH_USER) $(DEST_DIR)

clean:
	find data -mindepth 1 -not -name '.gitkeep' -delete

dry-run:
	$(PYTHON) cli.py pilot --dry-run --config $(CONFIG)

preflight:
	$(PYTHON) cli.py preflight --dry-run --config $(CONFIG)

offline-bundle:
	$(PYTHON) -m harness.offline_bundle --root .

# Control-host environmental prerequisites (NOT node config): ansible drives the
# deploy, docker builds the deception-responder image, and Python 3.11 builds the
# offline wheelhouse — it MUST equal the attacker node's Python (deadsnakes 3.11) or
# the cp311 dependency wheels will not install. Run once on the control host; then run
# the harness / `make offline-bundle` under Python 3.11 (e.g. `python3.11 -m venv`).
bootstrap-control:
	sudo apt-get update
	sudo apt-get install -y software-properties-common
	sudo add-apt-repository -y ppa:deadsnakes/ppa
	sudo apt-get update
	sudo apt-get install -y ansible docker.io python3.11 python3.11-venv

# Software provisioning of an ALREADY-provisioned testbed (VMs + dual-homed control
# host must exist; terraform apply is separate + gated). Builds/stages the images
# and offline bundle, runs site.yml behind temporary egress, re-seals, and runs the
# isolation preflight — failing closed if the testbed is not sealed. Run on control.
deploy:
	CONFIG=$(CONFIG) INVENTORY=infra/ansible/inventory.ini MODE=egress PYTHON=$(PYTHON) \
		bash scripts/deploy.sh

teardown:
	@echo "Teardown is gated: 'cd infra && terraform destroy' (removes all VMs + the isolated bridge)."
	@exit 1
