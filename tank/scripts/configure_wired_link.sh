#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-edge}"
IFACE="${SYNC_TANK_WIRED_IFACE:-eth0}"
CONNECTION_NAME="${SYNC_TANK_WIRED_CONNECTION:-sync-tank-display-link}"

case "$ROLE" in
  edge)
    WIRED_IP="${SYNC_TANK_EDGE_WIRED_IP:-TANK_ONE_WIRED_IP/24}"
    ;;
  display)
    WIRED_IP="${SYNC_TANK_DISPLAY_WIRED_IP:-SYNC_WIRED_IP/24}"
    ;;
  *)
    echo "Usage: $0 [edge|display]" >&2
    exit 2
    ;;
esac

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

if nmcli connection show "$CONNECTION_NAME" >/dev/null 2>&1; then
  nmcli connection modify "$CONNECTION_NAME" \
    connection.interface-name "$IFACE" \
    connection.autoconnect yes \
    ipv4.method manual \
    ipv4.addresses "$WIRED_IP" \
    ipv4.gateway "" \
    ipv4.dns "" \
    ipv4.never-default yes \
    ipv6.method disabled
else
  nmcli connection add \
    type ethernet \
    ifname "$IFACE" \
    con-name "$CONNECTION_NAME" \
    connection.autoconnect yes \
    ipv4.method manual \
    ipv4.addresses "$WIRED_IP" \
    ipv4.never-default yes \
    ipv6.method disabled
fi

nmcli connection up "$CONNECTION_NAME" || true
nmcli connection show "$CONNECTION_NAME"
