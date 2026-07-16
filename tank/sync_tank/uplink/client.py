from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests


class HubClient:
    def __init__(self, tank_id: str, config: dict[str, Any]):
        self.tank_id = tank_id
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.api_key = str(config.get("api_key", ""))
        self.mode = str(config.get("mode", "snapshot_push"))
        self.timeout = float(config.get("timeout_seconds", 5))
        self.streaming: set[str] = set()

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            state = "disabled"
        elif self.streaming:
            state = "streaming"
        else:
            state = "configured"
        return {"enabled": self.enabled, "state": state, "base_url": self.base_url, "mode": self.mode}

    def test(self) -> dict[str, Any]:
        if not self.enabled:
            return {"state": "disabled", "ok": False}
        try:
            response = requests.get(f"{self.base_url}/api/sync-tank/health", headers=self._headers(), timeout=self.timeout)
            if response.status_code in (401, 403):
                return {"state": "auth_failed", "ok": False, "status_code": response.status_code}
            return {"state": "connected" if response.status_code < 500 else "unreachable", "ok": response.ok, "status_code": response.status_code}
        except requests.RequestException as exc:
            return {"state": "unreachable", "ok": False, "error": str(exc)}

    def register_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/api/nodes/register", payload)

    def send_heartbeat(self, node_id: str, status: str = "online") -> dict[str, Any]:
        return self._post_json("/api/nodes/heartbeat", {"node_id": node_id, "status": status})

    def register_cameras(self, node_id: str, cameras: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post_json("/api/cameras/register", {"node_id": node_id, "cameras": cameras})

    def send_frame_base64(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/api/cameras/frame", payload)

    def send_frame(self, camera: dict[str, Any], frame: bytes, note: str = "") -> dict[str, Any]:
        if not self.enabled:
            return {"state": "disabled", "ok": False}
        files = {"frame": ("frame.jpg", frame, "image/jpeg")}
        data = {
            "tank_id": self.tank_id,
            "camera_id": camera["id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": camera.get("source_type", ""),
            "camera_name": camera.get("name", ""),
            "note": note,
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/sync-tank/frame",
                headers=self._headers(),
                data=data,
                files=files,
                timeout=self.timeout,
            )
            if response.status_code in (401, 403):
                return {"state": "auth_failed", "ok": False, "status_code": response.status_code}
            return {"state": "connected", "ok": response.ok, "status_code": response.status_code}
        except requests.RequestException as exc:
            return {"state": "unreachable", "ok": False, "error": str(exc)}

    def start_stream(self, camera: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"state": "disabled", "ok": False}
        payload = {
            "tank_id": self.tank_id,
            "camera_id": camera["id"],
            "camera_name": camera.get("name", ""),
            "source_type": camera.get("source_type", ""),
            "stream_url": camera.get("stream_url") or camera.get("snapshot_url"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/sync-tank/stream",
                headers={**self._headers(), "content-type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code in (401, 403):
                return {"state": "auth_failed", "ok": False, "status_code": response.status_code}
            if response.ok:
                self.streaming.add(camera["id"])
            return {"state": "streaming" if response.ok else "unreachable", "ok": response.ok, "status_code": response.status_code}
        except requests.RequestException as exc:
            return {"state": "unreachable", "ok": False, "error": str(exc)}

    def stop_stream(self, camera_id: str) -> dict[str, Any]:
        self.streaming.discard(camera_id)
        return {"state": "connected" if self.enabled else "disabled", "ok": True}

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"state": "disabled", "ok": False}
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                headers={**self._headers(), "content-type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code in (401, 403):
                return {"state": "auth_failed", "ok": False, "status_code": response.status_code}
            result: dict[str, Any] = {"state": "connected" if response.ok else "unreachable", "ok": response.ok, "status_code": response.status_code}
            try:
                result["response"] = response.json()
            except ValueError:
                result["response_text"] = response.text[:500]
            return result
        except requests.RequestException as exc:
            return {"state": "unreachable", "ok": False, "error": str(exc)}

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"authorization": f"Bearer {self.api_key}"}
