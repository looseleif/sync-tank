#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ask() {
  local prompt="$1"
  local default_value="${2:-}"
  local answer
  if [ -n "$default_value" ]; then
    read -r -p "$prompt [$default_value]: " answer
    printf '%s' "${answer:-$default_value}"
  else
    read -r -p "$prompt: " answer
    printf '%s' "$answer"
  fi
}

ask_yes_no() {
  local prompt="$1"
  local default_value="${2:-y}"
  local answer
  read -r -p "$prompt [$default_value]: " answer
  answer="${answer:-$default_value}"
  case "$answer" in
    y|Y|yes|YES|Yes) return 0 ;;
    *) return 1 ;;
  esac
}

require_number() {
  local value="$1"
  local name="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a zero-or-greater integer, got '$value'" >&2
    exit 2
  fi
}

install_system_packages() {
  echo "Installing Sync Tank system packages..."
  sudo apt-get update
  sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    v4l-utils \
    i2c-tools \
    python3-smbus \
    network-manager \
    curl
}

enable_i2c() {
  echo "Enabling Raspberry Pi I2C..."
  if command -v raspi-config >/dev/null 2>&1; then
    sudo raspi-config nonint do_i2c 0
  else
    echo "raspi-config not found; enable I2C manually in /boot/firmware/config.txt with: dtparam=i2c_arm=on" >&2
    return 1
  fi
}

echo "Sync Tank tank-node setup"
echo
echo "This script writes config/tank_identity.yaml, config/sync_tank.yaml, and local runtime setup."
echo "It can also install dependencies, enable I2C, configure the wired link, and install boot startup."
echo

node_id="$(ask "Pi/node ID" "tank-pi-001")"
node_label="$(ask "Pi/node label" "TANK NODE 1")"
tank_id="$(ask "Tank ID this Pi serves" "tank-main")"
tank_label="$(ask "Human tank label" "$tank_id")"
expected_nodes="$(ask "ESP32 floater node IDs, comma-separated" "tank-cam-001,tank-cam-002")"

reefscope_count="$(ask "Expected ReefScope USB cameras" "2")"
lighthouse_count="$(ask "Expected Lighthouse pan/tilt cameras" "1")"
reeflex_count="$(ask "Expected REEFLEX arms" "1")"
solid_feeders="$(ask "Expected solid feeders" "1")"
liquid_feeders="$(ask "Expected liquid feeders" "0")"
misc_feeders="$(ask "Expected misc feeders" "0")"

require_number "$reefscope_count" "ReefScope count"
require_number "$lighthouse_count" "Lighthouse count"
require_number "$reeflex_count" "REEFLEX count"
require_number "$solid_feeders" "Solid feeder count"
require_number "$liquid_feeders" "Liquid feeder count"
require_number "$misc_feeders" "Misc feeder count"

wired_edge_ip="$(ask "Tank Pi wired IP for display link" "TANK_ONE_WIRED_IP")"
display_pi_ip="$(ask "Display/organizer Pi wired IP" "SYNC_WIRED_IP")"
public_url="$(ask "Tank dashboard URL advertised to hub" "http://${wired_edge_ip}:8080")"
camera_service_url="$(ask "Tank camera/control URL advertised to hub" "http://${wired_edge_ip}:5050")"

install_deps_args=()
if ask_yes_no "Install Python requirements with pip" "y"; then
  install_deps_args+=(--install-deps)
fi

if ask_yes_no "Install system packages with apt (ffmpeg, v4l-utils, i2c-tools, etc.)" "y"; then
  install_system_packages
fi

if ask_yes_no "Enable Raspberry Pi I2C for PCA9685 servos" "y"; then
  enable_i2c || true
fi

configure_wired_args=()
if ask_yes_no "Configure eth0 static wired display link now" "n"; then
  configure_wired_args+=(--configure-wired)
fi

boot_mode="$(ask "Boot startup mode: cron, systemd, or none" "cron")"
case "$boot_mode" in
  cron|systemd|none) ;;
  *)
    echo "Invalid boot startup mode: $boot_mode" >&2
    exit 2
    ;;
esac

./scripts/install-tank.sh \
  --node-id "$node_id" \
  --label "$node_label" \
  --tank-id "$tank_id" \
  --tank-label "$tank_label" \
  --expected-nodes "$expected_nodes" \
  --reefscope-count "$reefscope_count" \
  --lighthouse-count "$lighthouse_count" \
  --reeflex-count "$reeflex_count" \
  --solid-feeders "$solid_feeders" \
  --liquid-feeders "$liquid_feeders" \
  --misc-feeders "$misc_feeders" \
  --wired-edge-ip "$wired_edge_ip" \
  --display-pi-ip "$display_pi_ip" \
  --public-url "$public_url" \
  --camera-service-url "$camera_service_url" \
  --install-boot "$boot_mode" \
  "${install_deps_args[@]}" \
  "${configure_wired_args[@]}"

echo
echo "Setup complete."
echo
echo "Recommended next checks:"
echo "  ./scripts/start-tank.sh"
echo "  ./scripts/self-test-cameras.sh --repair"
echo "  ./scripts/test-servos.sh 0 91"
echo
echo "If I2C was just enabled, reboot before servo testing:"
echo "  sudo reboot"
