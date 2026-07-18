#!/usr/bin/env bash
set -euo pipefail

APP_URL="${SYNC_TANK_APP_URL:-http://127.0.0.1:8765/}"
DISPLAY_VALUE="${DISPLAY:-:0}"
XAUTHORITY_VALUE="${XAUTHORITY:-/home/zero/.Xauthority}"
PROFILE_DIR="${SYNC_TANK_CHROMIUM_PROFILE:-/home/zero/.config/sync-tank-kiosk-chromium}"
OUTPUT="${SYNC_TANK_DISPLAY_OUTPUT:-HDMI-A-1}"
ROTATION="${SYNC_TANK_DISPLAY_ROTATION:-left}"
WINDOW_SIZE="1080,1920"

export DISPLAY="$DISPLAY_VALUE"
export XAUTHORITY="$XAUTHORITY_VALUE"

wait_for_display() {
  for _ in $(seq 1 30); do
    if command -v wlr-randr >/dev/null 2>&1 && wlr-randr >/dev/null 2>&1; then
      return 0
    fi
    if command -v xrandr >/dev/null 2>&1 && xrandr --query >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 0
}

connected_wlr_output() {
  if wlr-randr | grep -q "^${OUTPUT} "; then
    printf '%s\n' "$OUTPUT"
    return
  fi
  wlr-randr | awk '/^[A-Za-z0-9-]+ / { print $1; exit }'
}

connected_x_output() {
  if xrandr --query | grep -q "^${OUTPUT} connected"; then
    printf '%s\n' "$OUTPUT"
    return
  fi
  xrandr --query | awk '/ connected/ { print $1; exit }'
}

wait_for_display

if command -v wlr-randr >/dev/null 2>&1; then
  ACTIVE_OUTPUT="$(connected_wlr_output)"
  if [ -n "$ACTIVE_OUTPUT" ]; then
    if wlr-randr | awk -v output="$ACTIVE_OUTPUT" '
      $1 == output { in_output = 1; next }
      in_output && /^[A-Za-z0-9-]+ / { in_output = 0 }
      in_output && $1 == "1920x1080" { found = 1 }
      END { exit found ? 0 : 1 }
    '; then
      wlr-randr --output "$ACTIVE_OUTPUT" --mode 1920x1080@60.000000 --transform 270 || true
    else
      wlr-randr --output "$ACTIVE_OUTPUT" --transform 270 || true
    fi
  fi
elif command -v xrandr >/dev/null 2>&1; then
  ACTIVE_OUTPUT="$(connected_x_output)"
  if [ -n "$ACTIVE_OUTPUT" ]; then
    if xrandr --query | awk -v output="$ACTIVE_OUTPUT" '
      $1 == output { in_output = 1; next }
      in_output && /^[A-Za-z].* (connected|disconnected)/ { in_output = 0 }
      in_output && $1 == "1920x1080" { found = 1 }
      END { exit found ? 0 : 1 }
    '; then
      xrandr --output "$ACTIVE_OUTPUT" --mode 1920x1080 --rotate "$ROTATION" || true
    else
      CURRENT_MODE="$(xrandr --query | awk -v output="$ACTIVE_OUTPUT" '
        $1 == output { split($3, parts, "+"); print parts[1]; exit }
      ')"
      if [ -n "$CURRENT_MODE" ]; then
        xrandr --output "$ACTIVE_OUTPUT" --mode "$CURRENT_MODE" --rotate "$ROTATION" || true
        WIDTH="${CURRENT_MODE%x*}"
        HEIGHT="${CURRENT_MODE#*x}"
        WINDOW_SIZE="${HEIGHT},${WIDTH}"
      fi
    fi
  fi
fi

pkill -f 'chromium.*127.0.0.1:8765' >/dev/null 2>&1 || true

exec chromium \
  --app="$APP_URL" \
  --start-fullscreen \
  --window-position=0,0 \
  --window-size="$WINDOW_SIZE" \
  --user-data-dir="$PROFILE_DIR" \
  --password-store=basic \
  --no-first-run \
  --no-default-browser-check \
  --disable-session-crashed-bubble \
  --disable-infobars
