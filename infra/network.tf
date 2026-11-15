# Isolated, internal-only Linux bridge for the entire testbed.
#
# It has NO physical ports (no uplink) and no address/gateway on the host, so
# there is no route off the Proxmox node for anything attached to it: no external
# egress from the testbed or the attacker tooling. This is the primary containment
# control. (The Proxmox host management interface is separate and is NOT bridged
# here.) See infra/README.md for the containment model and the LLM-egress note.

resource "proxmox_network_linux_bridge" "isolated" {
  node_name = var.node_name
  name      = var.isolated_bridge
  comment   = "Isolated MTD testbed (no uplink / no external egress)"
  autostart = true
}
