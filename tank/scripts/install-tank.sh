#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'MSG'
Usage:
  ./scripts/install-tank.sh [options]

Configures this checkout for a Sync Tank Raspberry Pi tank node.

Common options:
  --node-id ID                 Default: tank-pi-001
  --label TEXT                 Default: TANK NODE
  --profile ID                 Default: tank1-lighthouse
  --tank-id ID                 Default: tank-main
  --tank-label TEXT            Default: tank ID
  --expected-nodes LIST        Default: tank-cam-001,tank-cam-002
  --reefscope-count N          Default: 2
  --lighthouse-count N         Default: 1
  --reeflex-count N            Default: 1
  --solid-feeders N            Default: 1
  --liquid-feeders N           Default: 0
  --misc-feeders N             Default: 0
  --public-url URL             Default: detected LAN IP on port 8080
  --camera-service-url URL     Default: detected LAN IP on port 5050
  --wired-edge-ip IP           Example: TANK_ONE_WIRED_IP or TANK_ONE_WIRED_IP/24
  --display-pi-ip IP           Example: SYNC_WIRED_IP
  --configure-wired            Configure eth0 as the edge wired display link
  --install-boot MODE          none, cron, or systemd. Default: none
  --install-deps               pip install --user -r requirements.txt
  -h, --help

Example:
  ./scripts/install-tank.sh \
    --node-id tank-pi-002 \
    --label "TANK NODE 2" \
    --tank-label "Main Reef" \
    --expected-nodes tank-cam-003,tank-cam-004 \
    --wired-edge-ip TANK_ONE_WIRED_IP \
    --install-boot cron
MSG
}

require_value() {
  if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

node_id="${SYNC_TANK_NODE_ID:-tank-pi-001}"
node_label="${SYNC_TANK_NODE_LABEL:-TANK NODE}"
profile_id="${SYNC_TANK_PROFILE_ID:-tank1-lighthouse}"
tank_id="${SYNC_TANK_TANK_ID:-tank-main}"
tank_label="${SYNC_TANK_TANK_LABEL:-}"
public_url="${SYNC_TANK_PUBLIC_URL:-}"
camera_service_url="${SYNC_TANK_CAMERA_SERVICE_URL:-}"
expected_nodes="${SYNC_TANK_EXPECTED_NODES:-tank-cam-001,tank-cam-002}"
reefscope_count="${SYNC_TANK_REEFSCOPE_COUNT:-2}"
lighthouse_count="${SYNC_TANK_LIGHTHOUSE_COUNT:-1}"
reeflex_count="${SYNC_TANK_REEFLEX_COUNT:-1}"
solid_feeders="${SYNC_TANK_SOLID_FEEDERS:-1}"
liquid_feeders="${SYNC_TANK_LIQUID_FEEDERS:-0}"
misc_feeders="${SYNC_TANK_MISC_FEEDERS:-0}"
wired_edge_ip="${SYNC_TANK_WIRED_EDGE_IP:-}"
display_pi_ip="${SYNC_TANK_DISPLAY_PI_IP:-SYNC_WIRED_IP}"
configure_wired="false"
install_boot="${SYNC_TANK_INSTALL_BOOT:-none}"
install_deps="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --node-id)
      require_value "$@"
      node_id="$2"
      shift 2
      ;;
    --label)
      require_value "$@"
      node_label="$2"
      shift 2
      ;;
    --profile)
      require_value "$@"
      profile_id="$2"
      shift 2
      ;;
    --tank-id)
      require_value "$@"
      tank_id="$2"
      shift 2
      ;;
    --tank-label)
      require_value "$@"
      tank_label="$2"
      shift 2
      ;;
    --public-url)
      require_value "$@"
      public_url="$2"
      shift 2
      ;;
    --camera-service-url)
      require_value "$@"
      camera_service_url="$2"
      shift 2
      ;;
    --expected-nodes)
      require_value "$@"
      expected_nodes="$2"
      shift 2
      ;;
    --reefscope-count)
      require_value "$@"
      reefscope_count="$2"
      shift 2
      ;;
    --lighthouse-count)
      require_value "$@"
      lighthouse_count="$2"
      shift 2
      ;;
    --reeflex-count)
      require_value "$@"
      reeflex_count="$2"
      shift 2
      ;;
    --solid-feeders)
      require_value "$@"
      solid_feeders="$2"
      shift 2
      ;;
    --liquid-feeders)
      require_value "$@"
      liquid_feeders="$2"
      shift 2
      ;;
    --misc-feeders)
      require_value "$@"
      misc_feeders="$2"
      shift 2
      ;;
    --wired-edge-ip)
      require_value "$@"
      wired_edge_ip="$2"
      shift 2
      ;;
    --display-pi-ip)
      require_value "$@"
      display_pi_ip="$2"
      shift 2
      ;;
    --configure-wired)
      configure_wired="true"
      shift
      ;;
    --install-boot)
      require_value "$@"
      install_boot="$2"
      shift 2
      ;;
    --install-deps)
      install_deps="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$install_boot" in
  none|cron|systemd) ;;
  *)
    echo "--install-boot must be one of: none, cron, systemd" >&2
    exit 2
    ;;
esac

cd "$(dirname "$0")/.."

if [ -z "$tank_label" ]; then
  tank_label="$tank_id"
fi

wired_host="${wired_edge_ip%%/*}"
if [ -n "$wired_host" ]; then
  if [ -z "$public_url" ]; then
    public_url="http://${wired_host}:8080"
  fi
  if [ -z "$camera_service_url" ]; then
    camera_service_url="http://${wired_host}:5050"
  fi
fi

if [ -z "$public_url" ]; then
  ip="$(hostname -I | awk '{print $1}')"
  public_url="http://${ip}:8080"
fi

if [ -z "$camera_service_url" ]; then
  ip="${public_url#http://}"
  ip="${ip%%:*}"
  camera_service_url="http://${ip}:5050"
fi

if [ "$install_deps" = "true" ]; then
  python3 -m pip install --user -r requirements.txt
fi

python3 - "$node_id" "$node_label" "$profile_id" "$tank_id" "$tank_label" "$public_url" "$camera_service_url" "$expected_nodes" "$reefscope_count" "$lighthouse_count" "$reeflex_count" "$solid_feeders" "$liquid_feeders" "$misc_feeders" "$wired_host" "$display_pi_ip" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def as_int(value: str, name: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc
    if number < 0:
        raise SystemExit(f"{name} must be zero or greater")
    return number


root = Path.cwd()
(
    node_id,
    node_label,
    profile_id,
    tank_id,
    tank_label,
    public_url,
    camera_service_url,
    expected_nodes_raw,
    reefscope_count_raw,
    lighthouse_count_raw,
    reeflex_count_raw,
    solid_feeders_raw,
    liquid_feeders_raw,
    misc_feeders_raw,
    wired_host,
    display_pi_ip,
) = sys.argv[1:]

expected_nodes = [item.strip() for item in expected_nodes_raw.split(",") if item.strip()]
reefscope_count = as_int(reefscope_count_raw, "reefscope-count")
lighthouse_count = as_int(lighthouse_count_raw, "lighthouse-count")
reeflex_count = as_int(reeflex_count_raw, "reeflex-count")
solid_feeders = as_int(solid_feeders_raw, "solid-feeders")
liquid_feeders = as_int(liquid_feeders_raw, "liquid-feeders")
misc_feeders = as_int(misc_feeders_raw, "misc-feeders")
role_split = {
    "lighthouse": lighthouse_count > 0,
    "reeflex": reeflex_count > 0,
    "note": "Tank node owns Lighthouse." if lighthouse_count and not reeflex_count else ("Tank node owns REEFLEX." if reeflex_count and not lighthouse_count else "Tank node owns configured Lighthouse/REEFLEX devices."),
}

config_path = root / "config" / "sync_tank.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
config["tank_id"] = tank_id

ingest = config.setdefault("ingest", {})
ingest["hub_id"] = node_id
ingest["host_id"] = node_id
ingest["host_label"] = node_label
ingest["public_url"] = public_url
ingest["camera_service_url"] = camera_service_url
ingest["expected_nodes"] = expected_nodes
ingest["allowed_nodes"] = expected_nodes
ingest["node_angles"] = {camera_id: index * 90 for index, camera_id in enumerate(expected_nodes)}
ingest.setdefault("usb_feed_allowed_cidrs", ["127.0.0.0/8", "REDACTED_PRIVATE_IP/24"])

config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

identity_path = root / "config" / "tank_identity.yaml"
identity = {
    "tank": {"id": tank_id, "label": tank_label},
    "node": {"id": node_id, "label": node_label, "role": "raspberry_pi_tank_node"},
    "network": {
        "wired_edge_ip": wired_host,
        "display_pi_ip": display_pi_ip,
        "public_url": public_url,
        "camera_service_url": camera_service_url,
        "usb_feed_allowed_cidrs": ingest.get("usb_feed_allowed_cidrs", ["127.0.0.0/8", "REDACTED_PRIVATE_IP/24"]),
    },
    "esp32": {
        "expected_nodes": expected_nodes,
        "allowed_nodes": expected_nodes,
        "node_angles": {camera_id: index * 90 for index, camera_id in enumerate(expected_nodes)},
    },
    "inventory": {
        "reefscope_cameras": reefscope_count,
        "lighthouse_cameras": lighthouse_count,
        "reeflex_arms": reeflex_count,
        "feeders": {"solid": solid_feeders, "liquid": liquid_feeders, "misc": misc_feeders},
    },
    "profile": {"id": profile_id, "role_split": role_split},
    "sync_node_contract": {
        "payload_shape": "pc-hub-payload-v1",
        "unique_fields": ["node_id", "tank_id", "camera_id"],
        "poll_endpoint": "/api/pc-hub/payload",
    },
}
identity_path.write_text(yaml.safe_dump(identity, sort_keys=False), encoding="utf-8")

for runtime_file in ("ingest_state.json", "cameras.json", "tank_layout.json"):
    path = root / "config" / runtime_file
    if path.exists():
        path.unlink()

node_config_path = root / "config" / "node_config.json"
node_config = {
    "node": {"id": node_id, "label": node_label, "role": "raspberry_pi_tank_node"},
    "inventory": {
        "robotic_arms": reeflex_count,
        "endoscope_cameras": reefscope_count,
        "floater_cameras": len(expected_nodes),
        "lighthouses": lighthouse_count,
    },
    "feeders": {"solid": solid_feeders, "liquid": liquid_feeders, "misc": misc_feeders},
    "feeder_viewports": {},
    "validation": {
        "status": "default",
        "validated_by_hand": False,
        "message": "Run the local dashboard and confirm the devices physically present on this tank.",
    },
    "profile": {"id": profile_id, "role_split": role_split},
    "cameras": {
        camera_id: {
            "label": f"Floater Camera #{index}",
            "camera_type": "floater_cam",
            "source_type": "esp32_upload",
            "node_id": node_id,
            "tank_id": tank_id,
            "enabled": True,
        }
        for index, camera_id in enumerate(expected_nodes, start=1)
    },
    "notes": "",
}
node_config_path.write_text(json.dumps(node_config, indent=2, sort_keys=True), encoding="utf-8")

print(f"Configured {node_label} ({node_id})")
print(f"Profile: {profile_id}")
print(f"Tank: {tank_label} ({tank_id})")
print(f"Public URL: {public_url}")
print(f"Camera service URL: {camera_service_url}")
print(f"Assigned ESP32 nodes: {', '.join(expected_nodes) or 'none'}")
print(f"Expected ReefScopes: {reefscope_count}")
print(f"Expected Lighthouse cameras: {lighthouse_count}")
print(f"Expected REEFLEX arms: {reeflex_count}")
print(f"Expected feeders: solid={solid_feeders}, liquid={liquid_feeders}, misc={misc_feeders}")
PY

chmod +x scripts/*.sh

if [ "$configure_wired" = "true" ]; then
  if [ -n "$wired_edge_ip" ]; then
    export SYNC_TANK_EDGE_WIRED_IP="$wired_edge_ip"
  fi
  ./scripts/configure_wired_link.sh edge
fi

install_cron_boot() {
  local boot_line="@reboot $PWD/scripts/start-tank.sh >> $PWD/logs/boot.log 2>&1"
  local tmp_file
  mkdir -p "$PWD/logs"
  tmp_file="$(mktemp)"
  crontab -l 2>/dev/null | grep -v "$PWD/scripts/start-tank.sh" > "$tmp_file" || true
  printf '%s\n' "$boot_line" >> "$tmp_file"
  crontab "$tmp_file"
  rm -f "$tmp_file"
}

case "$install_boot" in
  cron)
    install_cron_boot
    ;;
  systemd)
    ./scripts/install-systemd.sh
    ;;
  none)
    ;;
esac

cat <<MSG

Install complete.

Start everything now:
  cd $PWD
  ./scripts/start-tank.sh

Run the hardware self-test:
  cd $PWD
  ./scripts/self-test-cameras.sh --repair

Open the tank dashboard:
  ${public_url}

Boot startup:
  mode: ${install_boot}
MSG
