#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
from urllib import request, error


def detect_wlan_ip() -> str:
    interface = os.environ.get("SYNC_TANK_WIFI_INTERFACE", "wlan0")
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", interface],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)/", result.stdout)
        if match:
            return match.group(1)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "127.0.0.1"


def post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=5) as resp:
            resp.read()
    except error.URLError:
        pass


def camera_ids() -> list[str]:
    raw = os.environ.get("SYNC_TANK_CAMERA_IDS", "tank-cam-001,tank-cam-002,tank-cam-003,tank-cam-004")
    return [camera_id.strip() for camera_id in raw.split(",") if camera_id.strip()]


def camera_owner(camera_id: str, default_owner: str) -> str:
    owner_map = {
        "tank-cam-001": "tank-pi-001",
        "tank-cam-002": "tank-pi-001",
        "tank-cam-003": "tank-pi-002",
        "tank-cam-004": "tank-pi-002",
    }
    return os.environ.get(f"SYNC_TANK_OWNER_{camera_id.replace('-', '_').upper()}", owner_map.get(camera_id, default_owner))


def main() -> None:
    hub_base = os.environ.get("HUB_BASE", os.environ.get("SYNC_TANK_BASE_URL", "http://127.0.0.1:8765"))
    public_base = os.environ.get("PUBLIC_BASE", os.environ.get("SYNC_TANK_PUBLIC_BASE", f"http://{detect_wlan_ip()}"))
    public_base = public_base.rstrip("/")
    node_id = os.environ.get("SYNC_TANK_NODE_ID", "tank-pi-001")
    relay_node_id = os.environ.get("SYNC_TANK_RELAY_NODE_ID")
    hostname = os.environ.get("SYNC_TANK_HOSTNAME", "pi-zero")
    label = os.environ.get("SYNC_TANK_LABEL", "Tank Pi Zero")
    tank_id = os.environ.get("SYNC_TANK_TANK_ID", "tank-main")
    refresh_interval = int(os.environ.get("SYNC_TANK_REFRESH_INTERVAL", "30"))
    esp32_port = int(os.environ.get("SYNC_TANK_ESP32_PORT", "8080"))
    usb_port = int(os.environ.get("SYNC_TANK_USB_PORT", "5050"))

    node_payload = {
        "node_id": node_id,
        "node_type": "raspberry_pi_tank_node",
        "hostname": hostname,
        "label": label,
        "tank_ids": [tank_id],
        "lan_url": f"{public_base}:{esp32_port}",
        "status": "online",
    }
    cameras = []
    for camera_id in camera_ids():
        owner_id = camera_owner(camera_id, node_id)
        camera = {
            "camera_id": camera_id,
            "camera_type": "floater_cam",
            "source_type": "esp32_upload",
            "node_id": owner_id,
            "tank_id": tank_id,
            "latest_image_url": f"{public_base}:{esp32_port}/uploads/{camera_id}/latest.jpg",
            "status": "online",
        }
        if relay_node_id and relay_node_id != owner_id:
            camera["relay_node_id"] = relay_node_id
        cameras.append(camera)

    if os.environ.get("SYNC_TANK_REGISTER_USB", "0") == "1":
        cameras.append(
            {
                "camera_id": "usb_0",
                "camera_type": "endoscope_cam",
                "source_type": "usb_camera",
                "node_id": node_id,
                "tank_id": tank_id,
                "snapshot_url": f"{public_base}:{usb_port}/api/cameras/usb_0/snapshot",
                "stream_url": f"{public_base}:{usb_port}/api/cameras/usb_0/stream",
                "status": "online",
            }
        )

    camera_payload = {
        "node_id": node_id,
        "cameras": cameras,
    }

    while True:
        post_json(f"{hub_base}/api/nodes/register", node_payload)
        post_json(f"{hub_base}/api/cameras/register", camera_payload)
        post_json(f"{hub_base}/api/nodes/heartbeat", {"node_id": node_id, "status": "online"})
        time.sleep(refresh_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
