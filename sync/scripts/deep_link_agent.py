#!/usr/bin/env python3
import argparse
import json
import socket
import time
from urllib import error, request


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


def detect_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("PRIVATE_IP", 1))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def display_node_payload(display_node_id: str, display_base: str, label: str = "Pi Zero Display Node") -> dict:
    return {
        "node_id": display_node_id,
        "node_type": "sync_tank_display_node",
        "label": label,
        "lan_url": display_base.rstrip("/"),
        "status": "online",
    }


def build_deep_link_payload(
    layout: dict,
    display_node_id: str,
    display_base: str,
    label: str = "Pi Zero Display Node",
) -> dict:
    nodes = list(layout.get("nodes", []))
    cameras = list(layout.get("cameras", []))
    display_node = display_node_payload(display_node_id, display_base, label=label)
    all_nodes = [display_node, *nodes]

    linked_cameras = []
    for camera in cameras:
        linked = dict(camera)
        linked["display_node_id"] = display_node_id
        linked["deep_link_mode"] = "url_reference"
        linked_cameras.append(linked)

    return {
        "display_node": display_node,
        "nodes": all_nodes,
        "cameras": linked_cameras,
        "links": {
            "display_layout_url": f"{display_base.rstrip('/')}/api/layout",
            "display_dashboard_url": f"{display_base.rstrip('/')}/",
        },
        "capabilities": {
            "accepts_edge_inventory": True,
            "relays_stream_urls": True,
            "relays_raw_video": False,
            "runs_local_dashboard": True,
            "runs_tank_simulator": "planned",
        },
    }


def send_deep_link_payload(deep_link_base: str, payload: dict) -> dict:
    deep_link_base = deep_link_base.rstrip("/")
    display_node = payload["display_node"]
    results = {
        "display_node_registered": post_json(f"{deep_link_base}/api/nodes/register", display_node),
        "camera_inventory_registered": post_json(
            f"{deep_link_base}/api/cameras/register",
            {
                **display_node,
                "cameras": payload["cameras"],
                "replace": True,
            },
        ),
        "heartbeat_sent": post_json(
            f"{deep_link_base}/api/nodes/heartbeat",
            {
                "node_id": display_node["node_id"],
                "node_type": display_node["node_type"],
                "label": display_node["label"],
                "lan_url": display_node["lan_url"],
                "status": "online",
            },
        ),
    }
    post_json(f"{deep_link_base}/api/deep-link/register", payload)
    return results


def sync_deep_link(
    organizer_base: str,
    deep_link_base: str,
    display_base: str,
    display_node_id: str = "pi-zero-display-001",
    label: str = "Pi Zero Display Node",
) -> dict:
    layout = get_json(f"{organizer_base.rstrip('/')}/api/layout")
    if layout is None:
        return {"ok": False, "error": "organizer layout unavailable"}

    payload = build_deep_link_payload(layout, display_node_id, display_base, label=label)
    results = send_deep_link_payload(deep_link_base, payload)
    return {
        "ok": any(results.values()),
        "results": results,
        "payload": payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relay Pi display-node camera inventory to the Sync Tank main PC deep-link backend")
    parser.add_argument("--organizer-base", default="http://127.0.0.1:8765")
    parser.add_argument("--deep-link-base", required=True, help="main PC backend URL, for example http://PRIVATE_IP:8765")
    parser.add_argument("--display-base", default=None, help="display node URL visible to the main PC")
    parser.add_argument("--display-node-id", default="pi-zero-display-001")
    parser.add_argument("--label", default="Pi Zero Display Node")
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    display_base = args.display_base or f"http://{detect_lan_ip()}:8765"

    while True:
        result = sync_deep_link(
            organizer_base=args.organizer_base,
            deep_link_base=args.deep_link_base,
            display_base=display_base,
            display_node_id=args.display_node_id,
            label=args.label,
        )
        print(json.dumps({"ok": result["ok"], "results": result.get("results", {}), "sent_at": time.time()}), flush=True)
        if args.once:
            return 0 if result["ok"] else 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
