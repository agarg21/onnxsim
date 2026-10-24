#!/usr/bin/env bash
# Inside the guest: install the AXCL host package, register its driver with
# DKMS for every installed kernel, apply the host-driver-patches that still
# apply, load the stack, and check the card.
# Expects /mnt/share/axclhost_*.deb and /mnt/share/host-driver-patches/.
#
# The package version decides the source tree. 2.25.x ships a driver tree we copy
# and patch by hand; 3.x ships its own DKMS source (/usr/src/axclhost-<ver>), and
# the host-driver-patches are applied there only when they still apply (fix 7's
# slot_index_force is already in the 3.x source). Pairing matters: a 2.25 pac
# under the 3.6 runtime handshakes, then fails every command (0x80300909).
set -euo pipefail
DEB=$(ls /mnt/share/axclhost_*.deb | head -1)
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dkms build-essential patch python3 >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "linux-headers-$(uname -r)" >/dev/null || true
dpkg -i "$DEB" || apt-get install -f -y -qq
PKG=$(dpkg-query -W -f='${Version}' axclhost)
echo "axclhost package $PKG"
# every kernel the guest has (unattended upgrades install new ones; DKMS must
# build for each, not just the running one -- 2026-10-24)
KERNELS=$(ls /boot/vmlinuz-* | sed 's|/boot/vmlinuz-||' | sort -V)
for k in $KERNELS; do
  [ -d "/lib/modules/$k/build" ] || apt-get install -y -qq "linux-headers-$k" >/dev/null 2>&1 || true
done

case "$PKG" in
  2.*)
    DKMS_VER=2.25.0
    SRC=/usr/src/axcl-$DKMS_VER
    if [ ! -d "$SRC" ]; then
      mkdir -p "$SRC" && cp -r /usr/src/axcl/driver/axcl/. "$SRC"/
      cat > "$SRC/dkms.conf" <<CONF
PACKAGE_NAME="axcl"
PACKAGE_VERSION="$DKMS_VER"
AUTOINSTALL="yes"
CLEAN="make -C /lib/modules/\${kernelver}/build M=\${dkms_tree}/\${PACKAGE_NAME}/\${PACKAGE_VERSION}/build clean"
MAKE[0]="make -C /lib/modules/\${kernelver}/build M=\${dkms_tree}/\${PACKAGE_NAME}/\${PACKAGE_VERSION}/build -j\$(nproc) modules"
BUILT_MODULE_NAME[0]="ax_pcie_host_dev"
DEST_MODULE_LOCATION[0]="/updates/dkms"
BUILT_MODULE_NAME[1]="ax_pcie_mmb"
DEST_MODULE_LOCATION[1]="/updates/dkms"
BUILT_MODULE_NAME[2]="ax_pcie_msg"
DEST_MODULE_LOCATION[2]="/updates/dkms"
BUILT_MODULE_NAME[3]="axcl_host"
DEST_MODULE_LOCATION[3]="/updates/dkms"
CONF
    fi
    # the three pre-existing kernel-7 build edits (see host-driver-patches/NOTES.md)
    grep -q 'linux/vmalloc.h' "$SRC/axcl_pcie_host.c" || sed -i '0,/#include/s//#include <linux\/vmalloc.h>\n#include/' "$SRC/axcl_pcie_host.c"
    grep -q 'linux/vmalloc.h' "$SRC/ax_pcie_msg_usrdev.c" || sed -i '0,/#include/s//#include <linux\/vmalloc.h>\n#include/' "$SRC/ax_pcie_msg_usrdev.c"
    sed -i 's/SUPPORT_PCI_NET 1/SUPPORT_PCI_NET 0/' "$SRC/ax_pcie_dev.h"
    ;;
  3.*)
    DKMS_VER=$(ls -d /usr/src/axclhost-3.* | sed 's|.*/axclhost-||' | sort -V | tail -1)
    SRC=/usr/src/axclhost-$DKMS_VER
    [ -d "$SRC" ] || { echo "no DKMS source $SRC from the 3.x package" >&2; exit 1; }
    # dpkg -i just replaced the source with the pristine tree, so the slot-index fix
    # (driver fix 7) must go back in on every install. Without it the card reports
    # device id 3 but the driver expects bus 7: heartbeats dropped, "device 7: dead!".
    if grep -q slot_index_force "$SRC/drv/pcie/driver/host_dev/ax_pcie_dev_host.c"; then
      echo "slot_index_force already in $SRC"
    else
      patch -p1 -d "$SRC" < /mnt/share/axcl-3.6-slot-index.patch \
        || { echo "axcl-3.6-slot-index.patch does not apply to $SRC" >&2; exit 1; }
      echo "applied axcl-3.6-slot-index.patch"
    fi
    ;;
  *) echo "unknown axclhost version $PKG" >&2; exit 1 ;;
esac

# 2.x only: the host-driver-patches (fixes 4-7) were written against the 2.25 tree.
# They do not apply to the 3.x tree, which needs axcl-3.6-slot-index.patch above.
[ "${PKG%%.*}" = 2 ] && for p in /mnt/share/host-driver-patches/patches/*.patch; do
  name=$(basename "$p")
  if patch -p1 -d "$SRC" -R --dry-run -f < "$p" >/dev/null 2>&1; then
    echo "already applied: $name"
  elif patch -p1 -d "$SRC" --dry-run < "$p" >/dev/null 2>&1; then
    patch -p1 -d "$SRC" < "$p" && echo "applied: $name"
  else
    echo "not applicable to $DKMS_VER (skipped): $name"
  fi
done

case "$PKG" in 2.*) MOD=axcl/$DKMS_VER ;; *) MOD=axclhost/$DKMS_VER ;; esac
dkms add "$MOD" 2>/dev/null || true
for k in $KERNELS; do
  dkms build "$MOD" -k "$k" --force && dkms install "$MOD" -k "$k" --force
done
# The card reports a fixed device id (3); the driver otherwise derives the id
# from the PCI bus number, which differs in the guest -- see driver fix 7.
SLOT=${AXCL_SLOT_INDEX:-3}
echo 'options ax_pcie_host_dev slot_index_force='"$SLOT" > /etc/modprobe.d/axcl-guest.conf
modprobe -r axcl_host ax_pcie_mmb ax_pcie_msg ax_pcie_p2p_rc ax_pcie_host_dev 2>/dev/null || true
modprobe ax_pcie_host_dev && modprobe ax_pcie_msg && modprobe ax_pcie_mmb && modprobe axcl_host
echo "waiting for the handshake (up to ~2 min)..."
for i in $(seq 1 40); do
  dmesg | grep -q 'heartbeat_recv_thread, [0-9]*\]: target' && break
  sleep 3
done
sleep 20
timeout 30 axcl-smi || echo "axcl-smi did not answer yet; check dmesg for 'handshake'"
