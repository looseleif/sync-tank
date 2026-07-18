#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse


class EdgeReceiverApp:
    def __init__(self, storage_dir: str = "./edge_storage", allowed_hub_id: Optional[str] = None) -> None:
        self.storage_dir = Path(storage_dir)
        self.uploads_dir = self.storage_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.storage_dir / "state.json"
        self.allowed_hub_id = allowed_hub_id
        self.nodes: Dict[str, Dict] = {}
        self.commands: Dict[str, Dict] = {}
        self.events = []
        self._load_state()
        self._restore_latest_images()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.nodes = state.get("nodes", {})
        self.commands = state.get("commands", {})
        self.events = state.get("events", [])[-50:]

    def _save_state(self) -> None:
        state = {
            "nodes": self.nodes,
            "commands": self.commands,
            "events": self.events[-50:],
        }
        self.state_path.write_text(json.dumps(state, indent=2))

    def _restore_latest_images(self) -> None:
        for latest_path in self.uploads_dir.glob("*/latest.jpg"):
            node_id = latest_path.parent.name
            node = self.nodes.get(node_id, {"node_id": node_id, "node_type": "perimeter_camera_node"})
            node.update(
                {
                    "latest_image_url": f"/uploads/{node_id}/latest.jpg",
                    "latest_image_size_bytes": latest_path.stat().st_size,
                    "last_image_at": latest_path.stat().st_mtime,
                    "last_image_upload_status": node.get("last_image_upload_status", "restored"),
                    "status": node.get("status", "online"),
                }
            )
            self.nodes[node_id] = node

    def _record_event(self, kind: str, node_id: str, details: Optional[Dict] = None) -> None:
        self.events.append({"kind": kind, "node_id": node_id, "at": time.time(), "details": details or {}})
        self.events = self.events[-50:]

    def handle_heartbeat(self, payload: Dict) -> Dict:
        node_id = payload.get("node_id")
        if not node_id:
            raise ValueError("node_id is required")
        hub_id = payload.get("hub_id")
        if self.allowed_hub_id and hub_id and hub_id != self.allowed_hub_id:
            raise ValueError(f"node {node_id} belongs to hub {hub_id}, expected {self.allowed_hub_id}")

        heartbeat = dict(payload)
        heartbeat["last_heartbeat_at"] = time.time()
        self.nodes[node_id] = {**self.nodes.get(node_id, {}), **heartbeat}
        self._record_event("heartbeat", node_id, {"hub_id": hub_id, "status": heartbeat.get("status")})
        self._save_state()
        return {"ok": True, "node_id": node_id}

    def handle_image_upload(self, headers, image_bytes: bytes, fallback_node_id: Optional[str] = None) -> Dict:
        node_id = headers.get("X-Node-Id") or fallback_node_id
        if not node_id:
            raise ValueError("X-Node-Id header is required")
        hub_id = headers.get("X-Hub-Id")
        if self.allowed_hub_id and hub_id and hub_id != self.allowed_hub_id:
            raise ValueError(f"node {node_id} belongs to hub {hub_id}, expected {self.allowed_hub_id}")
        if not image_bytes.startswith(b"\xff\xd8"):
            raise ValueError("image body must be JPEG bytes")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        node_dir = self.uploads_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{node_id}_{timestamp}.jpg"
        image_path = node_dir / filename
        latest_path = node_dir / "latest.jpg"
        image_path.write_bytes(image_bytes)
        latest_path.write_bytes(image_bytes)

        node = self.nodes.get(node_id, {})
        node.update(
            {
                "node_id": node_id,
                "node_type": headers.get("X-Node-Type", node.get("node_type", "perimeter_camera_node")),
                "hub_id": hub_id or node.get("hub_id"),
                "firmware": headers.get("X-Firmware-Version", node.get("firmware")),
                "uptime_ms": _int_header(headers.get("X-Uptime-Ms")),
                "wifi_rssi": _int_header(headers.get("X-Wifi-Rssi")),
                "free_heap": _int_header(headers.get("X-Free-Heap")),
                "camera_available": True,
                "last_image_upload_status": "ok",
                "latest_image_url": f"/uploads/{node_id}/latest.jpg",
                "latest_image_filename": filename,
                "latest_image_size_bytes": len(image_bytes),
                "last_image_at": time.time(),
                "status": "online",
            }
        )
        self.nodes[node_id] = node
        self._record_event("image_upload", node_id, {"hub_id": hub_id, "size_bytes": len(image_bytes)})
        self._save_state()

        return {
            "ok": True,
            "node_id": node_id,
            "hub_id": hub_id,
            "latest_image": {
                "filename": filename,
                "url": f"/uploads/{node_id}/{filename}",
                "size_bytes": len(image_bytes),
            },
        }

    def set_command(self, node_id: str, command: Dict) -> Dict:
        if not node_id:
            raise ValueError("node_id is required")
        if not command.get("command"):
            raise ValueError("command is required")
        self.commands[node_id] = command
        self._record_event("command_set", node_id, command)
        self._save_state()
        return {"ok": True, "node_id": node_id, "command": command}

    def pop_command(self, node_id: str) -> Optional[Dict]:
        command = self.commands.pop(node_id, None)
        if command is not None:
            self._record_event("command_delivered", node_id, command)
            self._save_state()
        return command

    def get_upload(self, node_id: str, filename: str) -> Optional[bytes]:
        path = self.uploads_dir / node_id / filename
        if not path.exists() or not path.is_file():
            return None
        return path.read_bytes()

    def get_status(self) -> Dict:
        return {
            "status": "ok",
            "nodes": list(self.nodes.values()),
            "pending_commands": list(self.commands.keys()),
            "events": self.events[-20:],
        }


def _int_header(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


class EdgeReceiverHandler(BaseHTTPRequestHandler):
    server_version = "SyncTankEdgeReceiver/1.0"

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/uploads/"):
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self.send_error(HTTPStatus.NOT_FOUND, "upload not found")
                return
            _, node_id, filename = parts
            image = self.server.app.get_upload(node_id, filename)
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND, "upload not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.index_html().encode("utf-8"))
            return

        if path == "/api/status":
            self.send_json(self.server.app.get_status())
            return

        if path.startswith("/api/node/") and path.endswith("/command"):
            node_id = path.split("/")[3]
            command = self.server.app.pop_command(node_id)
            if command is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self.send_json(command)
            return

        if path.startswith("/uploads/"):
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self.send_error(HTTPStatus.NOT_FOUND, "upload not found")
                return
            _, node_id, filename = parts
            image = self.server.app.get_upload(node_id, filename)
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND, "upload not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""

        try:
            if path == "/api/node/heartbeat":
                payload = json.loads(body.decode("utf-8") or "{}")
                self.send_json(self.server.app.handle_heartbeat(payload))
                return

            if path == "/api/images/upload":
                query = parse_qs(parsed.query)
                fallback_node_id = query.get("node_id", [None])[0] or query.get("camera_id", [None])[0]
                self.send_json(self.server.app.handle_image_upload(self.headers, body, fallback_node_id=fallback_node_id))
                return

            if path.startswith("/api/node/") and path.endswith("/command"):
                node_id = path.split("/")[3]
                payload = json.loads(body.decode("utf-8") or "{}")
                self.send_json(self.server.app.set_command(node_id, payload))
                return
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def send_json(self, payload: Dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def index_html(self) -> str:
        return """<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Sync Tank Edge Receiver</title></head>
  <body>
    <h1>Sync Tank Edge Receiver</h1>
    <pre id=\"status\">Loading...</pre>
    <script>
      async function refresh() {
        const response = await fetch('/api/status', { cache: 'no-store' });
        document.getElementById('status').textContent = JSON.stringify(await response.json(), null, 2);
      }
      refresh();
      setInterval(refresh, 2000);
    </script>
  </body>
</html>"""


class EdgeReceiverHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, app: EdgeReceiverApp):
        super().__init__(server_address, handler_cls)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Tank ESP32 edge receiver")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--storage-dir", default=os.environ.get("SYNC_TANK_EDGE_STORAGE_DIR", "./edge_storage"))
    parser.add_argument("--allowed-hub-id", default=os.environ.get("SYNC_TANK_ALLOWED_HUB_ID"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = EdgeReceiverApp(storage_dir=args.storage_dir, allowed_hub_id=args.allowed_hub_id)
    server = EdgeReceiverHTTPServer((args.host, args.port), EdgeReceiverHandler, app)
    print(f"Sync Tank edge receiver listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
