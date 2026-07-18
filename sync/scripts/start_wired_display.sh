#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${SYNC_TANK_APP_DIR:-/home/zero/sync-tank}"
EDGE_BASE="${SYNC_TANK_EDGE_BASE:-http://TANK_ONE_WIRED_IP:8080}"
EDGE_BASES="${SYNC_TANK_EDGE_BASES:-${EDGE_BASE},http://TANK_TWO_WIRED_IP:8080}"
ORGANIZER_BASE="${SYNC_TANK_ORGANIZER_BASE:-http://127.0.0.1:8765}"
DEEP_LINK_BASE="${SYNC_TANK_DEEP_LINK_BASE:-${SYNC_TANK_HOME_COMPUTER_BASE:-}}"
PROCESSING_BASE="${SYNC_TANK_PROCESSING_BASE:-}"
DISPLAY_BASE="${SYNC_TANK_DISPLAY_BASE:-}"
LOG_DIR="${SYNC_TANK_LOG_DIR:-/home/zero/sync-tank/storage/logs}"

cd "$APP_DIR"
mkdir -p "$LOG_DIR"

pkill -f 'tank_manager.py.*--port 8765' >/dev/null 2>&1 || true
pkill -f 'scripts/edge_organizer_agent.py' >/dev/null 2>&1 || true
pkill -f 'scripts/feed_observer_agent.py' >/dev/null 2>&1 || true
pkill -f 'scripts/deep_link_agent.py' >/dev/null 2>&1 || true
sleep 0.5

nohup "$APP_DIR/.venv/bin/python" "$APP_DIR/tank_manager.py" --host 0.0.0.0 --port 8765 \
  >"$LOG_DIR/tank_manager.log" 2>&1 &
nohup "$APP_DIR/.venv/bin/python" "$APP_DIR/scripts/edge_organizer_agent.py" \
  --organizer-base "$ORGANIZER_BASE" \
  --edge-bases "$EDGE_BASES" \
  --seed-when-unreachable \
  --interval-seconds 5 \
  >"$LOG_DIR/edge_organizer_agent.log" 2>&1 &
OBSERVER_ARGS=(
  "$APP_DIR/scripts/feed_observer_agent.py"
  --organizer-base "$ORGANIZER_BASE"
  --storage-dir "$APP_DIR/storage"
  --interval-seconds 2
)
if [[ -n "$DEEP_LINK_BASE" ]]; then
  OBSERVER_ARGS+=(--deep-link-base "$DEEP_LINK_BASE")
fi
if [[ -n "$PROCESSING_BASE" ]]; then
  OBSERVER_ARGS+=(--processing-node-base "$PROCESSING_BASE")
fi
if [[ -n "$DISPLAY_BASE" ]]; then
  OBSERVER_ARGS+=(--display-base "$DISPLAY_BASE")
fi
nohup "$APP_DIR/.venv/bin/python" "${OBSERVER_ARGS[@]}" \
  >"$LOG_DIR/feed_observer_agent.log" 2>&1 &

if [[ -n "$DEEP_LINK_BASE" ]]; then
  DEEP_LINK_ARGS=(
    "$APP_DIR/scripts/deep_link_agent.py"
    --organizer-base "$ORGANIZER_BASE"
    --deep-link-base "$DEEP_LINK_BASE"
    --interval-seconds 15
  )
  if [[ -n "$DISPLAY_BASE" ]]; then
    DEEP_LINK_ARGS+=(--display-base "$DISPLAY_BASE")
  fi
  nohup "$APP_DIR/.venv/bin/python" "${DEEP_LINK_ARGS[@]}" \
    >"$LOG_DIR/deep_link_agent.log" 2>&1 &
fi

echo "Sync Tank display running at $ORGANIZER_BASE"
echo "Polling wired edge nodes at $EDGE_BASES"
if [[ -n "$DEEP_LINK_BASE" ]]; then
  echo "Forwarding observer events and display inventory to $DEEP_LINK_BASE"
fi
if [[ -n "$PROCESSING_BASE" ]]; then
  echo "Forwarding observer events to processing node $PROCESSING_BASE"
fi
echo "Launch the screen with: $APP_DIR/scripts/launch_display_app.sh"
