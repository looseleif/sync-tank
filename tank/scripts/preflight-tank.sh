#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Sync Tank preflight"
echo

echo "Project:"
echo "  $PWD"
echo

echo "Identity:"
PYTHONPATH="$PWD" python3 - <<'PY'
from sync_tank.config import load_config

config = load_config()
identity = config.raw.get("tank_identity") or {}
print(f"  profile: {config.raw.get('selected_profile') or identity.get('profile', {}).get('id') or 'custom'}")
print(f"  tank: {identity.get('tank', {}).get('label')} ({config.tank_id})")
print(f"  node: {identity.get('node', {}).get('label')} ({config.raw.get('ingest', {}).get('host_id')})")
print(f"  esp32: {', '.join(config.raw.get('ingest', {}).get('expected_nodes') or []) or 'none'}")
print(f"  rig devices: {', '.join(config.arm.get('devices', {})) or 'none'}")
PY
echo

echo "Required commands:"
for command in python3 pip3 ffmpeg v4l2-ctl i2cdetect curl; do
  if command -v "$command" >/dev/null 2>&1; then
    echo "  ok      $command -> $(command -v "$command")"
  else
    echo "  missing $command"
  fi
done
echo

echo "I2C:"
if ls /dev/i2c* >/dev/null 2>&1; then
  ls -l /dev/i2c* | sed 's/^/  /'
else
  echo "  no /dev/i2c* devices found"
fi
if command -v raspi-config >/dev/null 2>&1; then
  i2c_status="$(raspi-config nonint get_i2c 2>/dev/null || true)"
  echo "  raspi-config get_i2c: ${i2c_status:-unknown} (0 means enabled)"
fi
echo

echo "Video devices:"
if ls /dev/video* >/dev/null 2>&1; then
  ls -l /dev/video* | sed 's/^/  /'
else
  echo "  no /dev/video* devices found"
fi
echo

echo "HTTP services:"
for url in http://127.0.0.1:8080/api/pc-hub/payload http://127.0.0.1:5050/api/arm; do
  code="$(curl -s -o /dev/null -w '%{http_code}' "$url" || true)"
  echo "  $url -> $code"
done
