#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
IFACE="${SYNC_TANK_WIRED_IFACE:-eth0}"
DISPLAY_IP="${SYNC_TANK_DISPLAY_WIRED_IP:-SYNC_WIRED_IP}"
EDGE_IP="${SYNC_TANK_EDGE_WIRED_IP:-TANK_ONE_WIRED_IP}"
CIDR="${SYNC_TANK_WIRED_CIDR:-24}"

if [ "$ROLE" != "display" ] && [ "$ROLE" != "edge" ]; then
  echo "usage: $0 display|edge" >&2
  exit 2
fi

if [ "$ROLE" = "display" ]; then
  ADDRESS="${DISPLAY_IP}/${CIDR}"
else
  ADDRESS="${EDGE_IP}/${CIDR}"
fi

if ! command -v nmcli >/dev/null 2>&1; then
  echo "nmcli is required" >&2
  exit 1
fi

CONNECTION="$(nmcli -t -f NAME,DEVICE connection show | awk -F: -v iface="$IFACE" '$2 == iface { print $1; exit }')"
if [ -z "$CONNECTION" ]; then
  CONNECTION="sync-tank-wired-${ROLE}"
  sudo nmcli connection add type ethernet ifname "$IFACE" con-name "$CONNECTION"
fi

sudo nmcli connection modify "$CONNECTION" \
  ipv4.method manual \
  ipv4.addresses "$ADDRESS" \
  ipv4.never-default yes \
  ipv6.method disabled \
  connection.autoconnect yes

sudo nmcli connection up "$CONNECTION"

echo "Configured $IFACE as $ADDRESS for Sync Tank $ROLE wired link"
if [ "$ROLE" = "display" ]; then
  echo "Edge node should be reachable at http://${EDGE_IP}:8080 after it is configured."
else
  echo "Display node should be reachable at http://${DISPLAY_IP}:8765 after it is configured."
fi
