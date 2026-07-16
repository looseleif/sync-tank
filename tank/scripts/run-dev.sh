#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$PWD/scripts/process-utils.sh"

stop_port_processes 5050 "sync_tank.server" "Sync Tank camera/control"
stop_module_processes "sync_tank.server" "Sync Tank camera/control"
stop_stale_usb_stream_helpers

echo "Running USB camera startup self-test"
"$PWD/scripts/self-test-cameras.sh" --repair || true

export PYTHONPATH="$PWD"
python3 -m sync_tank.server
