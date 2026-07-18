#!/usr/bin/env python3
import argparse
import json
from urllib import error, request


def get_json(url: str, timeout_seconds: float = 3.0) -> dict | None:
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def fetch_sample(url: str, timeout_seconds: float = 3.0, max_bytes: int = 128_000) -> tuple[bool, int, str]:
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            data = response.read(max_bytes)
            return 200 <= response.status < 300 and bool(data), len(data), str(response.status)
    except error.HTTPError as exc:
        return False, 0, str(exc.code)
    except (OSError, error.URLError, TimeoutError) as exc:
        return False, 0, exc.__class__.__name__


def camera_payload(edge_base: str) -> dict:
    edge_base = edge_base.rstrip("/")
    for path in ("/api/pc-hub/payload", "/api/hub-payload", "/api/layout"):
        payload = get_json(f"{edge_base}{path}")
        if payload:
            return payload
    return {}


def extract_cameras(payload: dict) -> list[dict]:
    if "camera_registration" in payload:
        return payload.get("camera_registration", {}).get("cameras", [])
    return payload.get("cameras", [])


def absolutize(edge_base: str, url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"{edge_base.rstrip('/')}{url}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate USB video streams exposed by the wired Sync Tank edge node")
    parser.add_argument("--edge-base", default="http://TANK_ONE_WIRED_IP:8080")
    args = parser.parse_args()

    payload = camera_payload(args.edge_base)
    node = payload.get("node", payload)
    inventory = node.get("device_inventory") or payload.get("device_inventory") or {}
    cameras = extract_cameras(payload)
    usb_cameras = [
        camera for camera in cameras
        if camera.get("source_type") in ("usb", "usb_camera") or camera.get("camera_type") == "endoscope_cam"
    ]

    print(f"edge_base={args.edge_base}")
    print(f"inventory={json.dumps(inventory, sort_keys=True)}")
    print(f"registered_usb_endpoints={len(usb_cameras)}")
    for camera in usb_cameras:
        camera_id = camera.get("camera_id") or camera.get("id")
        snapshot_url = absolutize(args.edge_base, camera.get("snapshot_url") or f"/api/usb/{camera_id}/snapshot")
        stream_url = absolutize(args.edge_base, camera.get("stream_url") or f"/api/usb/{camera_id}/stream")
        snapshot_ok, snapshot_bytes, snapshot_status = fetch_sample(snapshot_url)
        stream_ok, stream_bytes, stream_status = fetch_sample(stream_url)
        print(
            f"{camera_id}: snapshot={snapshot_status}/{snapshot_bytes}B/{'ok' if snapshot_ok else 'fail'} "
            f"stream={stream_status}/{stream_bytes}B/{'ok' if stream_ok else 'fail'}"
        )

    expected_usb = sum(int((inventory.get("counts") or {}).get(key, 0)) for key in ("scope", "reeflex", "lighthouse"))
    if expected_usb and expected_usb != len(usb_cameras):
        print(f"warning=inventory_expects_{expected_usb}_usb_video_devices_but_registration_has_{len(usb_cameras)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
