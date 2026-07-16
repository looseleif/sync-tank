#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$PWD/scripts/process-utils.sh"

stop_port_processes 8080 "sync_tank.ingest" "Sync Tank ingest"
stop_module_processes "sync_tank.ingest" "Sync Tank ingest"

export PYTHONPATH="$PWD"
python3 -m sync_tank.ingest
