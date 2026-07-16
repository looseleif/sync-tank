#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

repair="False"
if [ "${1:-}" = "--repair" ]; then
  repair="True"
fi

python3 -c "import json; from sync_tank.cameras.usb import usb_camera_self_test; print(json.dumps(usb_camera_self_test(repair=${repair}), indent=2))"
