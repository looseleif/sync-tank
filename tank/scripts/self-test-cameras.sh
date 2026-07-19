#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
source "$PWD/scripts/python-runtime.sh"
python_bin="$(sync_tank_python "$PWD")"

repair="False"
if [ "${1:-}" = "--repair" ]; then
  repair="True"
fi

"$python_bin" -c "import json; from sync_tank.cameras.usb import usb_camera_self_test; print(json.dumps(usb_camera_self_test(repair=${repair}), indent=2))"
