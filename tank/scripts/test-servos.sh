#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

base_url="${SYNC_TANK_CAMERA_SERVICE_URL:-http://127.0.0.1:5050}"
channel="${1:-2}"
angle="${2:-91}"

echo "Sync Tank servo first-test"
echo "Service: $base_url"
echo "Requested channel: $channel"
echo "Requested angle: $angle"
echo

echo "Available I2C buses:"
if ls /dev/i2c* >/dev/null 2>&1; then
  ls -l /dev/i2c*
else
  echo "  none"
fi
echo

echo "Current arm status:"
status="$(curl -fsS "$base_url/api/arm")"
echo "$status"
echo

if printf '%s' "$status" | grep -q '"driver":"mock_'; then
  echo "Refusing movement test: servo driver is in mock fallback."
  echo "Enable I2C, confirm the PCA9685 is visible, then restart the camera/control service."
  exit 1
fi

echo "Sending safe single-channel command..."
curl -fsS \
  -H 'Content-Type: application/json' \
  -d "{\"channel\":${channel},\"angle\":${angle},\"device_id\":\"manual-test\",\"joint\":\"first-test\"}" \
  "$base_url/api/servo/channel"
echo
