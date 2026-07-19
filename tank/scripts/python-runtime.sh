#!/usr/bin/env bash

sync_tank_python() {
  local project_root="${1:-$PWD}"
  if [ -x "$project_root/.venv/bin/python" ]; then
    printf '%s\n' "$project_root/.venv/bin/python"
  else
    command -v python3
  fi
}
