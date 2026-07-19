#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

role="${1:-}"
case "$role" in
  tank1-raydar|tank2-reeflex) ;;
  *)
    echo "Usage: $0 tank1-raydar|tank2-reeflex" >&2
    exit 2
    ;;
esac

printf '%s\n' "$role" > config/node_role
echo "Selected $role for this Pi. This local selector is not committed."
echo "Restart with: sudo systemctl restart sync-tank.service sync-tank-ingest.service"
