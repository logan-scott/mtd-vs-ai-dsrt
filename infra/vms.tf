# Testbed VMs. Each role is provisioned from the shared cloud image onto the
# isolated bridge with a static IP and NO gateway (no external egress). Disks live
# on the vm_datastore (data2). Cloud-init handles the base user + keys; per-role
# software (vulnerable targets, offensive tooling, monitoring) is configured by the
# Ansible playbooks in infra/ansible/.

resource "proxmox_virtual_environment_vm" "node" {
  for_each = var.nodes

  node_name   = var.node_name
  name        = "mtd-${each.key}"
  description = "MTD testbed ${each.key} node (isolated, no egress)"
  tags        = each.value.tags

  # Disabled: the isolated cloud image has no qemu-guest-agent and cannot apt-install
  # one (no egress), so waiting for it just stalls each apply ~15 min. cloud-init and
  # SSH are unaffected; Proxmox uses ACPI for shutdown.
  agent {
    enabled = false
  }

  cpu {
    cores = each.value.cores
    type  = "x86-64-v2-AES"
  }

  memory {
    dedicated = each.value.memory_mb
  }

  disk {
    datastore_id = var.vm_datastore
    import_from  = proxmox_download_file.ubuntu_cloud.id
    interface    = "virtio0"
    size         = each.value.disk_gb
  }

  network_device {
    bridge = proxmox_network_linux_bridge.isolated.name
  }

  initialization {
    datastore_id = var.vm_datastore

    ip_config {
      ipv4 {
        # Static address, NO gateway => the node cannot route off the isolated net.
        address = "${var.isolated_subnet_prefix}.${each.value.ip_octet}/${var.isolated_cidr_prefix}"
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
