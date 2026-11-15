# Dual-homed control host for the pilot. This is the ONLY VM with external access.
# It runs the harness + attacker planner: it reaches the Anthropic API over the LAN
# leg (net1, on var.control_lan_bridge) and the testbed over vmbr9 (net0, static,
# NO gateway on that leg). Keep IP forwarding OFF in the guest so it never routes the
# testbed to the internet — the executor and testbed nodes stay egress-less. The
# ANTHROPIC_API_KEY .secrets file lives on THIS VM, not on the testbed nodes.
#
# net0 -> vmbr9  : static 172.31.66.<control_ip_octet>, no gateway (testbed access)
# net1 -> LAN    : static control_lan_address + gateway (default route -> internet)

resource "proxmox_virtual_environment_vm" "control" {
  node_name   = var.node_name
  name        = "mtd-control"
  description = "MTD pilot control host (harness + planner; dual-homed, forwarding off)"
  tags        = ["testbed", "control"]

  agent {
    enabled = false
  }

  cpu {
    cores = var.control_cores
    type  = "x86-64-v2-AES"
  }

  memory {
    dedicated = var.control_memory_mb
  }

  disk {
    datastore_id = var.vm_datastore
    import_from  = proxmox_download_file.ubuntu_cloud.id
    interface    = "virtio0"
    size         = var.control_disk_gb
  }

  # net0: isolated testbed leg.
  network_device {
    bridge = proxmox_network_linux_bridge.isolated.name
  }

  # net1: LAN leg for internet / Anthropic API.
  network_device {
    bridge = var.control_lan_bridge
  }

  initialization {
    datastore_id = var.vm_datastore

    # ip_config order MUST match network_device order: net0 (testbed) then net1 (LAN).
    ip_config {
      ipv4 {
        # Static, NO gateway on the isolated leg.
        address = "${var.isolated_subnet_prefix}.${var.control_ip_octet}/${var.isolated_cidr_prefix}"
      }
    }
    ip_config {
      ipv4 {
        # LAN leg carries the default route so the control host reaches the model API.
        address = var.control_lan_address
        gateway = var.control_lan_gateway
      }
    }

    user_account {
      username = var.ci_username
      keys     = var.ssh_public_keys
    }
  }

  operating_system {
    type = "l26"
  }
}
