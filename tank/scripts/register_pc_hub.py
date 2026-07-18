from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any

import requests

from sync_tank.ingest import create_ingest_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Register this Sync Tank edge node and all camera feeds with the PC hub.")
    parser.add_argument("pc_hub_url", help="PC hub base URL, for example http://PRIVATE_IP:8765")
    parser.add_argument("--api-key", default="", help="Optional PC hub bearer token")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    app = create_ingest_app()
    context = app.config["SYNC_TANK_INGEST"]
    store = context["store"]
    settings = context["settings"]
    edge_url = settings.public_url or f"http://{_lan_ip()}:{settings.port}"
    base_url = args.pc_hub_url.rstrip("/")
    headers = {"content-type": "application/json"}
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"

    node_payload = store.pc_node_payload(edge_url)
    camera_payload = {
        "node_id": settings.host_id,
        "cameras": store.pc_camera_inventory(edge_url),
    }
    heartbeat_payload = {"node_id": settings.host_id, "status": "online"}
    result: dict[str, Any] = {
        "edge_url": edge_url,
        "pc_hub_url": base_url,
        "camera_count": len(camera_payload["cameras"]),
        "camera_ids": [camera["camera_id"] for camera in camera_payload["cameras"]],
        "requests": {},
    }

    result["requests"]["register_node"] = _post(base_url, "/api/nodes/register", node_payload, headers, args.timeout)
    result["requests"]["register_cameras"] = _post(base_url, "/api/cameras/register", camera_payload, headers, args.timeout)
    result["requests"]["heartbeat"] = _post(base_url, "/api/nodes/heartbeat", heartbeat_payload, headers, args.timeout)
    result["requests"]["health"] = _get(base_url, "/api/cameras/health", headers, args.timeout)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["requests"]["register_cameras"].get("ok") else 1


def _post(base_url: str, path: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    try:
        response = requests.post(f"{base_url}{path}", headers=headers, json=payload, timeout=timeout)
        return _response_result(response)
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def _get(base_url: str, path: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url}{path}", headers={key: value for key, value in headers.items() if key != "content-type"}, timeout=timeout)
        return _response_result(response)
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def _response_result(response: requests.Response) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": response.ok, "status_code": response.status_code}
    try:
        result["response"] = response.json()
    except ValueError:
        result["response_text"] = response.text[:1000]
    return result


def _lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
