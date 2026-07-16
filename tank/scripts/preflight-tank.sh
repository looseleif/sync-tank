#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Sync Tank preflight"
echo

echo "Project:"
echo "  $PWD"
echo

echo "Identity:"
if [ -f config/tank_identity.yaml ]; then
  python3 - <<'PY'
from pathlib import Path
import yaml
identity = yaml.safe_load(Path("config/tank_identity.yaml").read_text()) or {}
print(f"  tank: {identity.get('tank', {}).get('label')} ({identity.get('tank', {}).get('id')})")
print(f"  node: {identity.get('node', {}).get('label')} ({identity.get('node', {}).get('id')})")
print(f"  public_url: {identity.get('network', {}).get('public_url')}")
print(f"  camera_service_url: {identity.get('network', {}).get('camera_service_url')}")
print(f"  esp32: {', '.join(identity.get('esp32', {}).get('expected_nodes') or []) or 'none'}")
PY
else
  echo "  missing config/tank_identity.yaml"
fi
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
