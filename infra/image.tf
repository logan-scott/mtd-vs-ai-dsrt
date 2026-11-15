# Download the pinned Ubuntu 22.04 LTS cloud image so VM disks can be created from
# it via import_from. Uses the 'import' content type because a disk's import_from
# source must be type 'images' or 'import' (an 'iso' file is rejected by VM create).
# Requirements this implies: image_datastore must be root-writable (use 'local', not
# a squashed network share) AND have the 'Import' content role enabled, and the
# stored file_name must end in .qcow2 (the 'import' role rejects a .img extension;
# the Ubuntu cloud .img is qcow2-format, so the rename is correct). The image is
# pulled once from the provisioning host; the testbed VMs never reach the internet.

resource "proxmox_download_file" "ubuntu_cloud" {
  content_type = "import"
  datastore_id = var.image_datastore
  node_name    = var.node_name
  url          = var.ubuntu_image_url
  file_name    = var.ubuntu_image_filename
  overwrite    = false
}
