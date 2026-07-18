#!/usr/bin/env python3
import argparse
import json
import time
from urllib import error, request


EDGE_BASE = "http://TANK_ONE_WIRED_IP:8080"
EDGE_BASES = ("http://TANK_ONE_WIRED_IP:8080", "http://TANK_TWO_WIRED_IP:8080")
CAMERA_SERVICE_BASE = "http://TANK_ONE_WIRED_IP:5050"


def known_edge_payload(edge_base: str = EDGE_BASE, status: str = "offline") -> dict:
    edge_base = edge_base.rstrip("/")
    camera_service_base = edge_base.replace(":8080", ":5050")
    is_tank_2 = edge_base.endswith(".12:8080") or "TANK_TWO_WIRED_IP" in edge_base
    node_id = "tank-pi-002" if is_tank_2 else "tank-pi-001"
    tank_id = "tank-2" if is_tank_2 else "tank-1"
    floater_ids = ("tank-cam-003", "tank-cam-004") if is_tank_2 else ("tank-cam-001", "tank-cam-002")
    return {
        "node_id": node_id,
        "node_type": "raspberry_pi_tank_node",
        "label": "EDGE NODE 2" if is_tank_2 else "EDGE NODE 1",
        "tank_ids": [tank_id],
        "lan_url": edge_base,
        "camera_service_url": camera_service_base,
        "status": status,
        "cameras": [
            {
                "camera_id": floater_ids[0],
                "camera_type": "floater_cam",
                "source_type": "esp32_upload",
                "node_id": node_id,
                "tank_id": tank_id,
                "latest_image_url": f"{edge_base}/uploads/{floater_ids[0]}/latest.jpg",
                "status": status,
            },
            {
                "camera_id": floater_ids[1],
                "camera_type": "floater_cam",
                "source_type": "esp32_upload",
                "node_id": node_id,
                "tank_id": tank_id,
                "latest_image_url": f"{edge_base}/uploads/{floater_ids[1]}/latest.jpg",
                "status": status,
            },
            {
                "camera_id": "usb_0",
                "camera_type": "endoscope_cam",
                "source_type": "usb_camera",
                "node_id": node_id,
                "tank_id": tank_id,
                "snapshot_url": f"{camera_service_base}/api/cameras/usb_0/snapshot",
                "stream_url": f"{camera_service_base}/api/cameras/usb_0/stream",
                "status": status,
            },
            {
                "camera_id": "usb_2",
                "camera_type": "endoscope_cam",
                "source_type": "usb_camera",
                "node_id": node_id,
                "tank_id": tank_id,
                "snapshot_url": f"{camera_service_base}/api/cameras/usb_2/snapshot",
                "stream_url": f"{camera_service_base}/api/cameras/usb_2/stream",
                "status": status,
            },
        ],
    }


def get_json(url: str, timeout_seconds: float = 3.0) -> dict | None:
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def post_json(url: str, payload: dict, timeout_seconds: float = 3.0) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response.read()
            return 200 <= response.status < 300
    except error.URLError:
        return False


def normalize_payload(payload: dict) -> dict:
    if "camera_registration" in payload:
        node = payload.get("node", {})
        registration = payload.get("camera_registration", {})
        return {**node, "cameras": registration.get("cameras", []), "node_id": registration.get("node_id", node.get("node_id"))}
    if "node" in payload and "cameras" in payload:
        return {**payload["node"], "cameras": payload["cameras"]}
    if "cameras" in payload:
        return payload
    return payload


def absolutize_urls(payload: dict, edge_base: str, camera_service_base: str) -> dict:
    edge_base = edge_base.rstrip("/")
    camera_service_base = camera_service_base.rstrip("/")
    for camera in payload.get("cameras", []):
        latest_image_url = camera.get("latest_image_url")
        if latest_image_url and latest_image_url.startswith("/"):
            camera["latest_image_url"] = f"{edge_base}{latest_image_url}"
        snapshot_url = camera.get("snapshot_url")
        if snapshot_url and snapshot_url.startswith("/"):
            camera["snapshot_url"] = f"{camera_service_base}{snapshot_url}"
        stream_url = camera.get("stream_url")
        if stream_url and stream_url.startswith("/"):
            camera["stream_url"] = f"{camera_service_base}{stream_url}"
    return payload


def apply_tank_role_overrides(payload: dict, edge_base: str) -> dict:
    edge_base = edge_base.rstrip("/")
    is_tank_2 = edge_base.endswith(".12:8080") or "TANK_TWO_WIRED_IP" in edge_base
    node_id = "tank-pi-002" if is_tank_2 else "tank-pi-001"
    tank_id = "tank-2" if is_tank_2 else "tank-1"

    payload["node_id"] = payload.get("node_id") or node_id
    if payload["node_id"] == node_id:
        payload["tank_ids"] = [tank_id]
        for camera in payload.get("cameras", []):
            raw_camera_id = camera.get("camera_id") or camera.get("id")
            camera["node_id"] = node_id
            camera["tank_id"] = tank_id
            if raw_camera_id and (raw_camera_id.startswith("usb_") or camera.get("source_type") in ("usb", "usb_camera")):
                camera["edge_camera_id"] = raw_camera_id
                camera["camera_id"] = f"{node_id}-{raw_camera_id}"
                camera["id"] = camera["camera_id"]
                camera["display_source_id"] = raw_camera_id
    return payload


def fetch_edge_payload(edge_base: str) -> dict | None:
    edge_base = edge_base.rstrip("/")
    camera_service_base = edge_base.replace(":8080", ":5050")
    for path in ("/api/pc-hub/payload", "/api/hub-payload"):
        payload = get_json(f"{edge_base.rstrip('/')}{path}")
        if payload:
            normalized = absolutize_urls(normalize_payload(payload), edge_base, camera_service_base)
            normalized = apply_tank_role_overrides(normalized, edge_base)
            if normalized.get("cameras"):
                return normalized
    return None


def split_edge_bases(edge_bases: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if not edge_bases:
        return [EDGE_BASE]
    if isinstance(edge_bases, str):
        candidates = edge_bases.split(",")
    else:
        candidates = []
        for entry in edge_bases:
            candidates.extend(str(entry).split(","))
    seen = set()
    bases = []
    for candidate in candidates:
        base = candidate.strip().rstrip("/")
        if base and base not in seen:
            seen.add(base)
            bases.append(base)
    return bases or [EDGE_BASE]


def run_once_for_edge(organizer_base: str, edge_base: str, seed_when_unreachable: bool) -> bool:
    payload = fetch_edge_payload(edge_base)
    if payload is None and seed_when_unreachable:
        payload = known_edge_payload(edge_base, status="offline")
    if payload is None:
        return False
    return post_json(f"{organizer_base.rstrip('/')}/api/cameras/register", payload)


def run_once(organizer_base: str, edge_bases: list[str], seed_when_unreachable: bool) -> bool:
    ok = False
    for edge_base in edge_bases:
        ok = run_once_for_edge(organizer_base, edge_base, seed_when_unreachable) or ok
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register Sync Tank edge-node camera inventory with the Pi Zero organizer")
    parser.add_argument("--organizer-base", default="http://127.0.0.1:8765")
    parser.add_argument("--edge-base", action="append", default=[])
    parser.add_argument("--edge-bases", default=",".join(EDGE_BASES))
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--seed-when-unreachable", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    edge_bases = split_edge_bases(args.edge_base or args.edge_bases)
    while True:
        run_once(args.organizer_base, edge_bases, args.seed_when_unreachable)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
