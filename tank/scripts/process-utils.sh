#!/usr/bin/env bash

stop_matching_pid() {
  local pid="$1"
  local label="$2"

  if [ -z "$pid" ] || [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ]; then
    return 0
  fi
  if [ ! -r "/proc/$pid/cmdline" ]; then
    return 0
  fi

  echo "Stopping stale $label process $pid"
  kill "$pid" 2>/dev/null || true
}

force_stop_if_alive() {
  local pid="$1"
  local label="$2"

  if [ -z "$pid" ] || [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ]; then
    return 0
  fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "Force stopping stale $label process $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_port_processes() {
  local port="$1"
  local match="$2"
  local label="$3"
  local pids
  local pid
  local cmdline
  local stopped=()

  if ! command -v fuser >/dev/null 2>&1; then
    return 0
  fi

  pids="$(fuser "${port}/tcp" 2>/dev/null || true)"
  for pid in $pids; do
    if [ ! -r "/proc/$pid/cmdline" ]; then
      continue
    fi
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    case "$cmdline" in
      *"$match"*)
        stop_matching_pid "$pid" "$label"
        stopped+=("$pid")
        ;;
      *)
        echo "Port $port is in use by non-Sync-Tank process $pid: $cmdline"
        echo "Not stopping it automatically."
        ;;
    esac
  done

  if [ "${#stopped[@]}" -gt 0 ]; then
    sleep 1
    for pid in "${stopped[@]}"; do
      force_stop_if_alive "$pid" "$label"
    done
  fi
}

stop_module_processes() {
  local module="$1"
  local label="$2"
  local pid
  local cmdline
  local stopped=()

  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    if [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ] || [ ! -r "$proc/cmdline" ]; then
      continue
    fi
    cmdline="$(tr '\0' ' ' < "$proc/cmdline")"
    case "$cmdline" in
      *"python"*"-m ${module}"*)
        stop_matching_pid "$pid" "$label"
        stopped+=("$pid")
        ;;
    esac
  done

  if [ "${#stopped[@]}" -gt 0 ]; then
    sleep 1
    for pid in "${stopped[@]}"; do
      force_stop_if_alive "$pid" "$label"
    done
  fi
}

stop_stale_usb_stream_helpers() {
  local pid
  local cmdline
  local stopped=()

  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    if [ "$pid" = "$$" ] || [ "$pid" = "${BASHPID:-}" ] || [ ! -r "$proc/cmdline" ]; then
      continue
    fi
    cmdline="$(tr '\0' ' ' < "$proc/cmdline")"
    case "$cmdline" in
      *"ffmpeg"*"-f v4l2"*"/dev/video"*)
        stop_matching_pid "$pid" "USB camera helper"
        stopped+=("$pid")
        ;;
    esac
  done

  if [ "${#stopped[@]}" -gt 0 ]; then
    sleep 1
    for pid in "${stopped[@]}"; do
      force_stop_if_alive "$pid" "USB camera helper"
    done
  fi
}
