#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${SYNC_TANK_APP_DIR:-/home/zero/sync-tank}"

"$APP_DIR/scripts/start_wired_display.sh"
sleep 1
exec "$APP_DIR/scripts/launch_display_app.sh"
