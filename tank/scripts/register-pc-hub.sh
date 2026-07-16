#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 http://PC_IP:8765 [--api-key TOKEN]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

python3 scripts/register_pc_hub.py "$@"
