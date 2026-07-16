from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CameraRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save({})

    def load(self) -> dict[str, dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, cameras: dict[str, dict[str, Any]]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(cameras, handle, indent=2, sort_keys=True)

    def upsert(self, camera: dict[str, Any]) -> dict[str, Any]:
        cameras = self.load()
        camera_id = str(camera["id"])
        existing = cameras.get(camera_id, {})
        merged = {**existing, **camera, "last_seen": utc_now()}
        cameras[camera_id] = merged
        self.save(cameras)
        return merged

    def get(self, camera_id: str) -> dict[str, Any] | None:
        return self.load().get(camera_id)

    def list(self) -> list[dict[str, Any]]:
        return list(self.load().values())

    def mark_status(self, camera_id: str, status: str) -> None:
        cameras = self.load()
        if camera_id in cameras:
            cameras[camera_id]["status"] = status
            cameras[camera_id]["last_seen"] = utc_now()
            self.save(cameras)
