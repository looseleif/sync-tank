#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

if ! pgrep -f "python3 -m sync_tank.ingest" >/dev/null 2>&1; then
  nohup "$PWD/scripts/run-ingest.sh" >> "$PWD/logs/ingest.log" 2>&1 &
fi

if ! pgrep -f "python3 -m sync_tank.server" >/dev/null 2>&1; then
  nohup "$PWD/scripts/run-dev.sh" >> "$PWD/logs/camera-control.log" 2>&1 &
fi

echo "Sync Tank tank node launch requested."
echo "Ingest log: $PWD/logs/ingest.log"
echo "Camera/control log: $PWD/logs/camera-control.log"
