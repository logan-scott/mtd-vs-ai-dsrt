# Testbed infrastructure (Proxmox, Terraform, Ansible)

Infrastructure as code for the study's isolated testbed on a Proxmox VE node (`pve2` at `10.5.0.200`
in the study's lab). Terraform (`bpg/proxmox`) creates an internal-only network and the VMs, and
Ansible configures each node.

## Containment model

All testbed and attacker nodes attach to one internal-only Linux bridge (`vmbr9`) with no physical
uplink and no host address or gateway, so nothing attached to it can route off the Proxmox node. Each
VM gets a static `172.31.66.0/24` address and no default gateway. The `mtd-control` host is the one
exception and is dual-homed by design: a testbed leg on `vmbr9` (`172.31.66.5`) and a LAN leg for
the model API, with IP forwarding off so that it never routes the testbed to the internet.

```
                 Proxmox node pve2 (10.5.0.200)
                 ┌────────────────────────────────────────────────────┐
   mgmt/SSH ───► │  vmbrX (mgmt, host-only)   provider + Ansible       │
                 │                                                      │
   LAN/API ────► │  vmbr0  ── mtd-control  10.5.0.50  (harness+planner) │
                 │              └─ also on vmbr9: 172.31.66.5 (fwd OFF) │
                 │  vmbr9  ── ISOLATED, NO UPLINK, NO GATEWAY ────────  │
                 │    ├── mtd-targets     172.31.66.10  (DVWA/JS/…)     │
                 │    ├── mtd-attacker    172.31.66.20  (tooling)       │
                 │    ├── mtd-mtd         172.31.66.30  (orchestr.)     │
                 │    └── mtd-monitoring  172.31.66.40  (agents/DB)     │
                 └────────────────────────────────────────────────────┘
                          (only mtd-control reaches the internet; fwd OFF)
```

The LLM planner runs on the control host, and the actions it selects execute from `mtd-attacker`,
which has no external egress. Before any live run, a fail-closed preflight (`harness/preflight.py`)
checks this isolation and aborts on any violation.

## Prerequisites

- A Proxmox VE node; the defaults are the study's node `pve2` at `https://10.5.0.200:8006/`. Set
  other values in `terraform.tfvars` and `secrets.auto.tfvars`.
- A scoped API token that can manage VMs, storage, and node networking, in a `secrets.auto.tfvars`
  copied from `secrets.auto.tfvars.example`.
- Storage: VM disks on `vm_datastore` (`data2` by default), with SSD or NVMe storage recommended for
  the targets VM; the cloud image on `software-images` with the `import` content type enabled; and a
  datastore with `snippets` enabled for cloud-init user data.
- Base image: the pinned Ubuntu 22.04 LTS cloud image (`variables.tf`).
- Host budget: about 12 vCPU, 48 GiB of memory, and 300 GiB of disk (the `nodes` variable).
- Terraform 1.6 or later, and Ansible on the control host.

## Provisioning

```bash
cd infra
cp secrets.auto.tfvars.example secrets.auto.tfvars
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Then, from the repository root on the control host:

```bash
make bootstrap-control
cp infra/ansible/inventory.example.ini infra/ansible/inventory.ini
make deploy CONFIG=harness/experiment_config.full.yaml
```

`make deploy` builds the local images and the attacker node's offline bundle, opens a temporary
egress window to install packages and pull the public target images, runs `site.yml`, and then
closes the window and verifies that the testbed is sealed. No trial can start until that check
passes. `terraform destroy` removes the VMs and the isolated bridge.

## Files

| File | Purpose |
|---|---|
| `versions.tf` | Terraform and the pinned `bpg/proxmox` provider |
| `providers.tf` | Provider configuration (token from variables or the environment) |
| `variables.tf` | All inputs (placement, network, sizing, image) |
| `network.tf` | The isolated bridge with no uplink (`vmbr9`) |
| `image.tf` | Import of the pinned Ubuntu 22.04 cloud image |
| `vms.tf` | Testbed VMs (cloud-init, static IP, no gateway) |
| `outputs.tf` | Bridge name, node IPs, VM IDs |
| `*.tfvars.example` | Templates for the gitignored secret and non-secret variables |
| `ansible/` | Post-provisioning configuration of each node, and `reconfigure.yml` |
