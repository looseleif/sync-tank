#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sudo install -m 0644 systemd/sync-tank.service /etc/systemd/system/sync-tank.service
sudo install -m 0644 systemd/sync-tank-ingest.service /etc/systemd/system/sync-tank-ingest.service
sudo systemctl daemon-reload
sudo systemctl enable sync-tank.service sync-tank-ingest.service

cat <<'MSG'
Installed Sync Tank services.

Start or restart now:
  sudo systemctl restart sync-tank.service sync-tank-ingest.service

Check status:
  systemctl status sync-tank.service sync-tank-ingest.service
MSG
