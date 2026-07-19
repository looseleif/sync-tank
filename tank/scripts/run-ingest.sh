#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$PWD/scripts/process-utils.sh"
source "$PWD/scripts/python-runtime.sh"
python_bin="$(sync_tank_python "$PWD")"

stop_port_processes 8080 "sync_tank.ingest" "Sync Tank ingest"
stop_module_processes "sync_tank.ingest" "Sync Tank ingest"

export PYTHONPATH="$PWD"
"$python_bin" -m sync_tank.ingest
