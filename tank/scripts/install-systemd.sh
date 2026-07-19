#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

service_user="$(id -un)"
service_root="$PWD"
temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

for service in sync-tank.service sync-tank-ingest.service; do
  sed \
    -e "s|@SYNC_TANK_USER@|$service_user|g" \
    -e "s|@SYNC_TANK_ROOT@|$service_root|g" \
    "systemd/$service" > "$temp_dir/$service"
  sudo install -m 0644 "$temp_dir/$service" "/etc/systemd/system/$service"
done
sudo systemctl daemon-reload
sudo systemctl enable sync-tank.service sync-tank-ingest.service

cat <<'MSG'
Installed Sync Tank services.

Start or restart now:
  sudo systemctl restart sync-tank.service sync-tank-ingest.service

Check status:
  systemctl status sync-tank.service sync-tank-ingest.service
MSG
