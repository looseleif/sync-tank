#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

out_dir="${1:-/tmp}"
mkdir -p "$out_dir"
archive="$out_dir/sync-tank-node-$(date +%Y%m%d-%H%M%S).tar.gz"

tar \
  --exclude='./.venv' \
  --exclude='./__pycache__' \
  --exclude='./**/__pycache__' \
  --exclude='./*.pyc' \
  --exclude='./.pytest_cache' \
  --exclude='./test_uploads' \
  --exclude='./logs' \
  --exclude='./logs/*' \
  --exclude='./*.log' \
  --exclude='./config/ingest_state.json' \
  --exclude='./config/cameras.json' \
  --exclude='./config/tank_layout.json' \
  --exclude='./ingest.log' \
  --exclude='./.lgd-*' \
  -czf "$archive" \
  .

echo "$archive"
