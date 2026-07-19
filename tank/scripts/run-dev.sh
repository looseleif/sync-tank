#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$PWD/scripts/process-utils.sh"
source "$PWD/scripts/python-runtime.sh"
python_bin="$(sync_tank_python "$PWD")"

stop_port_processes 5050 "sync_tank.server" "Sync Tank camera/control"
stop_module_processes "sync_tank.server" "Sync Tank camera/control"
stop_stale_usb_stream_helpers

echo "Running USB camera startup self-test"
"$PWD/scripts/self-test-cameras.sh" --repair || true

export PYTHONPATH="$PWD"
"$python_bin" -m sync_tank.server
