#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

service_user="$(id -un)"
service_root="$PWD"
temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

echo "Preparing isolated Python runtime..."
if [ ! -x "$service_root/.venv/bin/python" ]; then
  if ! python3 -m venv --system-site-packages "$service_root/.venv"; then
    echo "Unable to create $service_root/.venv. Install python3-venv, then rerun this script." >&2
    exit 1
  fi
fi
"$service_root/.venv/bin/python" -m pip install --disable-pip-version-check -r "$service_root/requirements.txt"
"$service_root/.venv/bin/python" -c 'import flask, flask_cors, requests, yaml, gpiozero, smbus2'

legacy_boot_line="$service_root/scripts/start-tank.sh"
if crontab -l 2>/dev/null | grep -Fq "$legacy_boot_line"; then
  echo "Removing legacy cron boot launcher for this checkout..."
  crontab -l 2>/dev/null | grep -Fv "$legacy_boot_line" > "$temp_dir/crontab" || true
  crontab "$temp_dir/crontab"
fi

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

Runtime:
  .venv created and requirements verified
  legacy cron launcher removed when present

Start or restart now:
  sudo systemctl restart sync-tank.service sync-tank-ingest.service

Check status:
  systemctl status sync-tank.service sync-tank-ingest.service
MSG
