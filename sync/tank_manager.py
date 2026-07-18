#!/usr/bin/env python3
import argparse
import base64
import binascii
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from urllib import error as urlerror, request as urlrequest

from wildlife_system import AI_DISCLOSURE, SIGHTING_LABELS, VisionController, ask_the_deep, is_jpeg, normalize_structure, select_best_capture


CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".mjs": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
}

DEFAULT_TANK_DIMENSIONS = {"x": 25, "y": 25, "z": 25, "unit": "in"}
NODE_STALE_SECONDS = float(os.environ.get("SYNC_TANK_NODE_STALE_SECONDS", "20"))


class TankManagerApp:
    def __init__(self, storage_dir: str = "./storage", openai_transport=None, control_transport=None) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.storage_dir / "layout.json"
        self.nodes: Dict[str, Dict] = {}
        self.cameras: Dict[str, Dict] = {}
        self.tanks: Dict[str, Dict] = {}
        self.scene_items: Dict[str, Dict] = {}
        self.detections: List[Dict] = []
        self.observations: Dict[str, Dict] = {}
        self.organisms: Dict[str, Dict] = {}
        self.sightings: Dict[str, Dict] = {}
        self.capture_locks: set[str] = set()
        self.capture_guard = threading.Lock()
        self.auto_capture_at: Dict[str, float] = {}
        self.openai_transport = openai_transport
        self.frame_dir = self.storage_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = self.storage_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.sightings_dir = self.storage_dir / "sightings"
        self.sightings_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self.vision = VisionController(
            self.get_control_url,
            post_json=control_transport,
            frame_source=self._vision_frame,
            capture_callback=self.capture_sighting,
        )

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.nodes = state.get("nodes", {})
        self.cameras = state.get("cameras", {})
        self.tanks = state.get("tanks", {})
        self.scene_items = state.get("scene_items", {})
        self.detections = state.get("detections", [])
        self.observations = state.get("observations", {})
        self.organisms = state.get("organisms", {})
        self.sightings = state.get("sightings", {})
        self._clear_unsaved_setup_state()
        self._save_state()

    def _clear_unsaved_setup_state(self) -> None:
        node_setup_saved = any(
            (node.get("setup_state") or {}).get("validated_by_hand")
            for node in self.nodes.values()
        )
        saved_tank_ids = {
            tank_id
            for tank_id, tank in self.tanks.items()
            if tank.get("saved_by_user") or (tank.get("hardware_validated") and node_setup_saved)
        }
        if not saved_tank_ids:
            self.scene_items = {
                item_id: item for item_id, item in self.scene_items.items()
                if item.get("item_type") == "structure_shape"
            }
        for tank_id, tank in self.tanks.items():
            if tank_id in saved_tank_ids:
                continue
            tank["dimensions"] = dict(DEFAULT_TANK_DIMENSIONS)
            tank["setup_complete"] = False
            tank["hardware_validated"] = False
        for camera in self.cameras.values():
            tank_id = camera.get("tank_id")
            if tank_id and tank_id in saved_tank_ids:
                continue
            camera["placement"] = {
                "placed": False,
                "position": None,
                "target": None,
                "fov_degrees": camera.get("placement", {}).get("fov_degrees", 70 if camera.get("source_type") == "usb_camera" else 60),
            }

    def _save_state(self) -> None:
        state = {
            "nodes": self.nodes,
            "cameras": self.cameras,
            "tanks": self.tanks,
            "scene_items": self.scene_items,
            "detections": self.detections,
            "observations": self.observations,
            "organisms": self.organisms,
            "sightings": self.sightings,
        }
        self.state_path.write_text(json.dumps(state, indent=2))

    def register_node(self, node: Dict) -> Dict:
        self._apply_display_role_contract_to_node(node)
        node_id = node["node_id"]
        registered = {**self.nodes.get(node_id, {}), **node}
        registered["last_seen_at"] = time.time()
        self.nodes[node_id] = registered
        self._save_state()
        return registered

    def _apply_display_role_contract_to_node(self, node: Dict) -> None:
        node_id = node.get("node_id")
        tank_ids = node.get("tank_ids") or []
        is_tank_1 = node_id == "tank-pi-001" or "tank-1" in tank_ids
        is_tank_2 = node_id == "tank-pi-002" or "tank-2" in tank_ids
        control_urls = node.get("control_urls")
        if isinstance(control_urls, dict):
            node["control_urls"] = dict(control_urls)
        inventory = node.get("device_inventory")
        if isinstance(inventory, dict):
            counts = inventory.get("counts")
            if isinstance(counts, dict):
                inventory["owned_device_count"] = sum(int(value or 0) for value in counts.values())
        catalog = node.get("device_catalog")
        if isinstance(catalog, list):
            node["device_catalog"] = list(catalog)

    def _apply_display_role_contract_to_camera(self, camera: Dict) -> None:
        return

    def register_cameras(self, cameras: List[Dict]) -> List[Dict]:
        registered = []
        for camera in cameras:
            camera_id = camera.get("camera_id")
            if not camera_id:
                continue
            self._apply_display_role_contract_to_camera(camera)
            existing = self.cameras.get(camera_id, {})
            registered_camera = {**existing, **camera}
            self._apply_display_role_contract_to_camera(registered_camera)
            if existing.get("placement") and not camera.get("placement"):
                registered_camera["placement"] = existing["placement"]
            if existing.get("role_locked") and not camera.get("force_role_update"):
                for key in ("camera_type", "label", "name", "device_id", "role_locked", "assigned_role"):
                    if key in existing:
                        registered_camera[key] = existing[key]
            registered_camera["last_seen_at"] = time.time()
            self.cameras[camera_id] = registered_camera
            registered.append(registered_camera)
        if registered:
            self._save_state()
        return registered

    def register_camera_payload(self, payload: Dict) -> Dict:
        reporting_node_id = payload.get("node_id")
        node_fields = {
            key: payload[key]
            for key in (
                "node_id",
                "node_type",
                "hostname",
                "label",
                "tank_ids",
                "lan_url",
                "camera_service_url",
                "status",
                "active_tank_node_count",
                "control_urls",
                "device_catalog",
                "device_inventory",
                "setup_state",
            )
            if key in payload
        }
        if node_fields.get("node_id"):
            self.register_node(node_fields)
        incoming_cameras = payload.get("cameras", [])
        incoming_ids = {camera.get("camera_id") for camera in incoming_cameras if camera.get("camera_id")}
        if payload.get("replace", True) and incoming_ids:
            for camera_id in list(self.cameras.keys()):
                camera = self.cameras.get(camera_id, {})
                camera_node_id = camera.get("node_id") or camera.get("hub_id")
                same_reporting_node = reporting_node_id and camera_node_id == reporting_node_id
                if same_reporting_node and camera_id not in incoming_ids:
                    del self.cameras[camera_id]
        cameras = self.register_cameras(incoming_cameras)
        if incoming_ids:
            self._save_state()
        return {"status": "ok", "registered_cameras": len(cameras)}

    def _tank_label(self, tank_id: str) -> str:
        words = tank_id.replace("_", "-").split("-")
        return " ".join(word.capitalize() for word in words if word) or "Tank"

    def _default_tank(self, tank_id: str) -> Dict:
        return {
            "tank_id": tank_id,
            "label": self._tank_label(tank_id),
            "dimensions": dict(DEFAULT_TANK_DIMENSIONS),
            "setup_complete": False,
        }

    def _connected_tank_ids(self) -> List[str]:
        tank_ids = []
        for node in self.nodes.values():
            for tank_id in node.get("tank_ids") or []:
                if tank_id and tank_id not in tank_ids:
                    tank_ids.append(tank_id)
        for camera in self.cameras.values():
            tank_id = camera.get("tank_id")
            if tank_id and tank_id not in tank_ids:
                tank_ids.append(tank_id)
        return tank_ids

    def _refresh_node_activity(self) -> None:
        now = time.time()
        active_node_ids = set()
        changed = False
        for node_id, node in self.nodes.items():
            last_seen = float(node.get("last_seen_at") or 0)
            age = max(0, now - last_seen) if last_seen else None
            active = bool(last_seen and age <= NODE_STALE_SECONDS and node.get("status") != "offline")
            next_status = "online" if active else "offline"
            node["active"] = active
            node["last_seen_age_seconds"] = round(age, 1) if age is not None else None
            node["stale_after_seconds"] = NODE_STALE_SECONDS
            if node.get("status") != next_status:
                node["status"] = next_status
                changed = True
            if active:
                active_node_ids.add(node_id)

        for camera in self.cameras.values():
            node_id = camera.get("node_id") or camera.get("hub_id")
            if node_id and node_id in self.nodes:
                next_status = "online" if node_id in active_node_ids else "offline"
                if camera.get("status") != next_status:
                    camera["status"] = next_status
                    changed = True
                camera["node_active"] = node_id in active_node_ids

        if changed:
            self._save_state()

    def _reconcile_tanks(self) -> None:
        connected_tank_ids = self._connected_tank_ids()
        if not self.tanks:
            first_tank_id = connected_tank_ids[0] if connected_tank_ids else "main-tank"
            self.tanks[first_tank_id] = self._default_tank(first_tank_id)

        default_only = set(self.tanks.keys()) == {"main-tank"}
        if default_only and connected_tank_ids and connected_tank_ids[0] != "main-tank":
            connected_tank_id = connected_tank_ids[0]
            default_tank = self.tanks.pop("main-tank")
            self.tanks[connected_tank_id] = {
                **self._default_tank(connected_tank_id),
                **default_tank,
                "tank_id": connected_tank_id,
                "label": default_tank.get("label") or self._tank_label(connected_tank_id),
            }
            for item in self.scene_items.values():
                if item.get("tank_id") == "main-tank":
                    item["tank_id"] = connected_tank_id

        for tank_id in connected_tank_ids:
            if tank_id not in self.tanks:
                self.tanks[tank_id] = self._default_tank(tank_id)

    def get_layout(self) -> Dict:
        changed_marker = json.dumps(
            {"tanks": self.tanks, "scene_items": self.scene_items},
            sort_keys=True,
            default=str,
        )
        self._reconcile_tanks()
        next_marker = json.dumps(
            {"tanks": self.tanks, "scene_items": self.scene_items},
            sort_keys=True,
            default=str,
        )
        if next_marker != changed_marker:
            self._save_state()
        self._refresh_node_activity()
        return {
            "tanks": list(self.tanks.values()),
            "nodes": list(self.nodes.values()),
            "cameras": list(self.cameras.values()),
            "scene_items": list(self.scene_items.values()),
            "detections": self.detections,
            "observations": list(self.observations.values()),
            "organisms": list(self.organisms.values()),
            "sightings": self.list_sightings(),
            "vision": self.vision.status(),
            "health": self.get_health(),
        }

    def get_health(self) -> Dict:
        self._refresh_node_activity()
        now = time.time()
        online_nodes = [node for node in self.nodes.values() if node.get("active")]
        online_cameras = [camera for camera in self.cameras.values() if camera.get("status") == "online"]
        return {
            "status": "online",
            "updated_at": now,
            "nodes_online": len(online_nodes),
            "nodes_total": len(self.nodes),
            "cameras_online": len(online_cameras),
            "cameras_total": len(self.cameras),
            "node_stale_seconds": NODE_STALE_SECONDS,
            "active_node_ids": [node.get("node_id") for node in online_nodes],
        }

    def active_nodes_payload(self) -> Dict:
        self._refresh_node_activity()
        active_nodes = [node for node in self.nodes.values() if node.get("active")]
        return {
            "active_nodes": active_nodes,
            "active_node_ids": [node.get("node_id") for node in active_nodes],
            "nodes_total": len(self.nodes),
            "active_count": len(active_nodes),
            "stale_after_seconds": NODE_STALE_SECONDS,
            "updated_at": time.time(),
        }

    def update_layout(self, payload: Dict) -> Dict:
        if payload.get("replace_tanks"):
            self.tanks = {}
        if payload.get("replace_scene_items"):
            self.scene_items = {}
        if payload.get("clear_observations"):
            self.observations = {}
            self.detections = []

        for item_id in payload.get("delete_scene_item_ids", []):
            self.scene_items.pop(item_id, None)

        for camera_id in payload.get("delete_camera_ids", []):
            camera = self.cameras.get(camera_id)
            if camera:
                camera["placement"] = {"placed": False, "position": None, "target": None, "fov_degrees": camera.get("placement", {}).get("fov_degrees", 70)}
                camera["hidden_from_layout"] = True

        for tank in payload.get("tanks", []):
            tank_id = tank.get("tank_id") or tank.get("id")
            if tank_id:
                self.tanks[tank_id] = {**self.tanks.get(tank_id, {}), **tank, "tank_id": tank_id}

        for camera in payload.get("cameras", []):
            camera_id = camera.get("camera_id") or camera.get("id")
            if not camera_id:
                continue
            existing = self.cameras.get(camera_id, {"camera_id": camera_id})
            update = {**camera, "camera_id": camera_id, "last_seen_at": time.time()}
            self.cameras[camera_id] = {**existing, **update}

        for item in payload.get("scene_items", []):
            item_id = item.get("item_id") or item.get("id")
            if item_id:
                if item.get("item_type") == "structure_shape":
                    item = normalize_structure(item)
                self.scene_items[item_id] = {**self.scene_items.get(item_id, {}), **item, "item_id": item_id}

        if "detections" in payload:
            self.detections = payload.get("detections") or []

        self._save_state()
        return {"status": "ok", "layout": self.get_layout()}

    def register_observation(self, payload: Dict) -> Dict:
        observation_id = payload.get("observation_id") or f"obs-{int(time.time() * 1000)}"
        observation = {**self.observations.get(observation_id, {}), **payload, "observation_id": observation_id}
        observation["updated_at"] = time.time()
        self.observations[observation_id] = observation
        self.detections = [item for item in self.detections if item.get("observation_id") != observation_id]
        self.detections.insert(0, observation)
        self.detections = self.detections[:50]
        self._save_state()
        return observation

    def label_observation(self, payload: Dict) -> Dict:
        observation_id = payload.get("observation_id")
        label = payload.get("label") or payload.get("organism_name")
        if not observation_id or not label:
            raise ValueError("observation_id and label are required")
        observation = self.observations.get(observation_id, {"observation_id": observation_id})
        observation["label"] = label
        observation["identity_status"] = "labeled"
        observation["labeled_at"] = time.time()
        self.observations[observation_id] = observation
        organism_id = payload.get("organism_id") or label.lower().replace(" ", "-")
        self.organisms[organism_id] = {
            **self.organisms.get(organism_id, {}),
            "organism_id": organism_id,
            "label": label,
            "last_observation_id": observation_id,
            "last_seen_at": time.time(),
        }
        self.detections = [observation if item.get("observation_id") == observation_id else item for item in self.detections]
        self._save_state()
        return observation

    def handle_frame_upload(self, payload: Dict) -> Dict:
        camera_id = payload.get("camera_id")
        if not camera_id:
            raise ValueError("camera_id is required")

        camera = self.cameras.get(camera_id, {})
        image_bytes = payload.get("image_bytes")
        if not image_bytes and payload.get("image_base64"):
            image_bytes = base64.b64decode(payload.get("image_base64", ""))
        if not image_bytes:
            raise ValueError("image payload is required")

        frame_path = self.frame_dir / f"{camera_id}.jpg"
        upload_path = self.uploads_dir / camera_id / "latest.jpg"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(image_bytes)
        upload_path.write_bytes(image_bytes)

        camera["latest_image_url"] = f"/uploads/{camera_id}/latest.jpg"
        camera["last_frame_path"] = str(frame_path)
        camera["last_frame_at"] = time.time()
        camera["status"] = payload.get("status", camera.get("status", "online"))
        self.cameras[camera_id] = camera
        self._save_state()
        return {"status": "ok", "camera_id": camera_id}

    def get_snapshot_bytes(self, camera_id: str) -> Optional[bytes]:
        candidates = [
            self.frame_dir / f"{camera_id}.jpg",
            self.uploads_dir / camera_id / "latest.jpg",
        ]
        for path in candidates:
            if path.exists():
                return path.read_bytes()
        return None

    def get_camera(self, camera_id: str) -> Dict:
        return self.cameras.get(camera_id, {})

    def _remote_snapshot_bytes(self, camera: Dict) -> Optional[bytes]:
        remote_url = camera.get("snapshot_url") or camera.get("latest_image_url")
        if not remote_url:
            return None
        with urlrequest.urlopen(remote_url, timeout=4) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        return data

    def _vision_frame(self, camera_id: str) -> Optional[bytes]:
        image = self.get_snapshot_bytes(camera_id)
        if image:
            return image
        camera = self.cameras.get(camera_id) or {}
        return self._remote_snapshot_bytes(camera)

    def capture_sighting(self, payload: Dict) -> Dict:
        camera_id = str(payload.get("camera_id") or "")
        if not camera_id or camera_id not in self.cameras:
            raise ValueError("a known camera_id is required")
        trigger = str(payload.get("trigger") or "manual")
        if trigger not in ("manual", "raydar-auto"):
            raise ValueError("trigger must be manual or raydar-auto")
        now = time.time()
        if trigger == "raydar-auto" and now - self.auto_capture_at.get(camera_id, 0) < 30:
            raise ValueError("automatic capture cooldown is active")
        with self.capture_guard:
            if camera_id in self.capture_locks:
                raise ValueError("capture already in progress for this camera")
            self.capture_locks.add(camera_id)
        try:
            image = None
            burst_choice = select_best_capture(payload.get("burst") or [])
            if burst_choice and burst_choice.get("image_base64"):
                image = base64.b64decode(burst_choice["image_base64"], validate=True)
            if payload.get("image_base64"):
                image = base64.b64decode(payload["image_base64"], validate=True)
            image = image or self.get_snapshot_bytes(camera_id)
            if image is None:
                try:
                    image = self._remote_snapshot_bytes(self.cameras[camera_id])
                except (urlerror.URLError, TimeoutError, OSError):
                    image = None
            if not image or not is_jpeg(image):
                raise ValueError("camera does not currently have a valid JPEG")
            sighting_id = "sighting-" + uuid.uuid4().hex[:16]
            image_path = self.sightings_dir / f"{sighting_id}.jpg"
            image_path.write_bytes(image)
            label = payload.get("label", "Unknown")
            if label not in SIGHTING_LABELS:
                raise ValueError("invalid sighting label")
            camera = self.cameras[camera_id]
            sighting = {
                "sighting_id": sighting_id,
                "camera_id": camera_id,
                "tank_id": payload.get("tank_id") or camera.get("tank_id"),
                "timestamp": now,
                "trigger": trigger,
                "focus_region": payload.get("focus_region"),
                "scores": (burst_choice or {}).get("scores") or payload.get("scores") or {},
                "crop_url": payload.get("crop_url"),
                "image_url": f"/api/sightings/{sighting_id}/image",
                "label": label,
                "favorite": bool(payload.get("favorite", False)),
                "ai_field_note": None,
            }
            self.sightings[sighting_id] = sighting
            if trigger == "raydar-auto":
                self.auto_capture_at[camera_id] = now
            self._save_state()
            self.cleanup_sightings()
            return sighting
        finally:
            with self.capture_guard:
                self.capture_locks.discard(camera_id)

    def list_sightings(self) -> List[Dict]:
        return sorted(self.sightings.values(), key=lambda item: item.get("timestamp", 0), reverse=True)

    def cleanup_sightings(self, max_count: int = 500) -> int:
        removed = 0
        candidates = sorted(
            (item for item in self.sightings.values() if not item.get("favorite") and item.get("label", "Unknown") == "Unknown"),
            key=lambda item: item.get("timestamp", 0),
        )
        while len(self.sightings) > max_count and candidates:
            item = candidates.pop(0)
            sighting_id = item["sighting_id"]
            self.sightings.pop(sighting_id, None)
            path = self.sightings_dir / f"{sighting_id}.jpg"
            if path.exists():
                path.unlink()
            removed += 1
        if removed:
            self._save_state()
        return removed

    def sighting_image(self, sighting_id: str) -> Optional[bytes]:
        if sighting_id not in self.sightings:
            return None
        path = self.sightings_dir / f"{sighting_id}.jpg"
        return path.read_bytes() if path.exists() else None

    def analyze_sighting(self, sighting_id: str, payload: Dict) -> Dict:
        sighting = self.sightings.get(sighting_id)
        if not sighting:
            raise ValueError("unknown sighting")
        if payload.get("confirmed") is not True:
            raise ValueError("explicit image confirmation is required")
        image = self.sighting_image(sighting_id)
        if not image:
            raise ValueError("sighting image is unavailable")
        note = ask_the_deep(image, persona=payload.get("persona"), transport=self.openai_transport)
        sighting["ai_field_note"] = {**note, "created_at": time.time(), "disclosure": AI_DISCLOSURE}
        self._save_state()
        return sighting

    def update_sighting(self, sighting_id: str, payload: Dict) -> Dict:
        sighting = self.sightings.get(sighting_id)
        if not sighting:
            raise ValueError("unknown sighting")
        if "label" in payload:
            if payload["label"] not in SIGHTING_LABELS:
                raise ValueError("invalid sighting label")
            sighting["label"] = payload["label"]
        if "favorite" in payload:
            sighting["favorite"] = bool(payload["favorite"])
        self._save_state()
        return sighting

    def get_control_url(self, *keys: str, node_id: Optional[str] = None, tank_id: Optional[str] = None) -> Optional[str]:
        self._refresh_node_activity()
        ordered_nodes = [
            *[node for node in self.nodes.values() if node.get("active")],
            *[node for node in self.nodes.values() if not node.get("active")],
        ]
        for node in ordered_nodes:
            if node_id and node.get("node_id") != node_id:
                continue
            if tank_id and tank_id not in (node.get("tank_ids") or []):
                continue
            control_urls = node.get("control_urls") or {}
            for key in keys:
                url = control_urls.get(key)
                if url:
                    return url
        return None


class TankManagerHandler(BaseHTTPRequestHandler):
    server_version = "TankManager/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/layout":
            self.send_json(self.server.app.get_layout())
            return

        if path == "/api/health":
            self.send_json(self.server.app.get_health())
            return

        if path == "/api/nodes/active":
            self.send_json(self.server.app.active_nodes_payload())
            return

        if path == "/api/vision/status":
            self.send_json(self.server.app.vision.status())
            return

        if path == "/api/sightings":
            self.send_json({
                "sightings": self.server.app.list_sightings(),
                "labels": list(SIGHTING_LABELS),
                "ask_the_deep": {
                    "enabled": bool(os.environ.get("OPENAI_API_KEY")),
                    "disclosure": AI_DISCLOSURE,
                },
            })
            return

        if path.startswith("/api/sightings/") and path.endswith("/image"):
            sighting_id = path.split("/")[3]
            image = self.server.app.sighting_image(sighting_id)
            if image is None:
                self.send_error(HTTPStatus.NOT_FOUND, "sighting image not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "private, max-age=31536000, immutable")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)
            return

        if path == "/api/controls/arm":
            params = parse_qs(parsed.query)
            self.proxy_control_get(
                ("arm_status", "reeflex_status"),
                node_id=(params.get("node_id") or [None])[0],
                tank_id=(params.get("tank_id") or [None])[0],
            )
            return

        if path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.index_html().encode("utf-8"))
            return

        if path.startswith("/static/"):
            self.send_static(path)
            return

        if path.startswith("/uploads/"):
            camera_id = path.split("/")[-2]
            snapshot = self.server.app.get_snapshot_bytes(camera_id)
            if snapshot is None:
                self.send_error(HTTPStatus.NOT_FOUND, "snapshot not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("ETag", f'"{hashlib.sha256(snapshot).hexdigest()}"')
            camera = self.server.app.get_camera(camera_id)
            if camera.get("last_frame_at"):
                self.send_header("Last-Modified", self.date_time_string(camera["last_frame_at"]))
            self.end_headers()
            self.wfile.write(snapshot)
            return

        if path.startswith("/observer_events/"):
            target = (self.server.app.storage_dir / path.lstrip("/")).resolve()
            root = (self.server.app.storage_dir / "observer_events").resolve()
            if root not in target.parents or not target.exists() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "event frame not found")
                return
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/api/cameras/") and path.endswith("/snapshot"):
            camera_id = path.split("/")[3]
            snapshot = self.server.app.get_snapshot_bytes(camera_id)
            if snapshot is None:
                camera = self.server.app.get_camera(camera_id)
                remote_url = camera.get("snapshot_url") or camera.get("latest_image_url")
                if remote_url:
                    self.proxy_remote_snapshot(remote_url)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "snapshot not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("ETag", f'"{hashlib.sha256(snapshot).hexdigest()}"')
            camera = self.server.app.get_camera(camera_id)
            if camera.get("last_frame_at"):
                self.send_header("Last-Modified", self.date_time_string(camera["last_frame_at"]))
            self.end_headers()
            self.wfile.write(snapshot)
            return

        if path.startswith("/api/cameras/") and path.endswith("/stream"):
            camera_id = path.split("/")[3]
            camera = self.server.app.get_camera(camera_id)
            if camera.get("stream_url"):
                self.proxy_remote_stream(camera["stream_url"])
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            for _ in range(10):
                snapshot = self.server.app.get_snapshot_bytes(camera_id)
                if snapshot:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(snapshot)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(snapshot)
                    self.wfile.write(b"\r\n")
                time.sleep(0.5)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        if length > 10 * 1024 * 1024:
            self.send_json({"status": "error", "error": "request is too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        body = self.rfile.read(length).decode("utf-8") if length else ""

        if path == "/api/nodes/register":
            payload = json.loads(body or "{}")
            self.server.app.register_node(payload)
            self.send_json({"status": "ok"})
            return

        if path == "/api/nodes/heartbeat":
            payload = json.loads(body or "{}")
            node_id = payload.get("node_id")
            if node_id:
                node = self.server.app.nodes.get(node_id, {"node_id": node_id})
                node.update(payload)
                node["status"] = payload.get("status", "online")
                self.server.app.register_node(node)
            self.send_json({"status": "ok"})
            return

        if path == "/api/cameras/register":
            payload = json.loads(body or "{}")
            self.send_json(self.server.app.register_camera_payload(payload))
            return

        if path == "/api/layout":
            payload = json.loads(body or "{}")
            self.send_json(self.server.app.update_layout(payload))
            return

        if path == "/api/observations/register":
            payload = json.loads(body or "{}")
            self.send_json({"status": "ok", "observation": self.server.app.register_observation(payload)})
            return

        if path == "/api/observations/label":
            payload = json.loads(body or "{}")
            self.send_json({"status": "ok", "observation": self.server.app.label_observation(payload)})
            return

        if path == "/api/cameras/frame":
            payload = json.loads(body or "{}")
            self.server.app.handle_frame_upload(payload)
            self.send_json({"status": "ok"})
            return

        if path == "/api/esp32/upload":
            payload = json.loads(body or "{}")
            self.server.app.handle_frame_upload(payload)
            self.send_json({"status": "ok"})
            return

        if path == "/api/vision/raydar/start":
            self._vision_action(lambda payload: self.server.app.vision.start_raydar({**payload, "background": True}), body)
            return

        if path == "/api/vision/raydar/stop":
            self._vision_action(self.server.app.vision.stop_raydar, body)
            return

        if path == "/api/vision/reeflex/start":
            self._vision_action(self.server.app.vision.start_reeflex, body)
            return

        if path == "/api/vision/reeflex/stop":
            self._vision_action(self.server.app.vision.stop_reeflex, body)
            return

        if path == "/api/sightings/capture":
            try:
                sighting = self.server.app.capture_sighting(json.loads(body or "{}"))
                self.send_json({"status": "ok", "sighting": sighting}, HTTPStatus.CREATED)
            except (ValueError, TypeError, binascii.Error) as exc:
                self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if path.startswith("/api/sightings/") and path.endswith("/analyze"):
            sighting_id = path.split("/")[3]
            try:
                sighting = self.server.app.analyze_sighting(sighting_id, json.loads(body or "{}"))
                self.send_json({"status": "ok", "sighting": sighting})
            except ValueError as exc:
                self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self.send_json({"status": "disabled", "error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except (urlerror.URLError, TimeoutError, OSError) as exc:
                self.send_json({"status": "offline", "error": f"Ask the Deep unavailable: {exc}"}, HTTPStatus.BAD_GATEWAY)
            return

        if path.startswith("/api/sightings/") and len(path.strip("/").split("/")) == 3:
            sighting_id = path.split("/")[3]
            try:
                sighting = self.server.app.update_sighting(sighting_id, json.loads(body or "{}"))
                self.send_json({"status": "ok", "sighting": sighting})
            except ValueError as exc:
                self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/controls/lighthouse/pose":
            payload = json.loads(body or "{}")
            self.server.app.vision.set_manual("raydar")
            self.proxy_control_post(("lighthouse_pose",), payload)
            return

        if path == "/api/controls/reeflex/pose":
            payload = json.loads(body or "{}")
            self.server.app.vision.set_manual("reeflex")
            self.proxy_control_post(("reeflex_pose",), payload)
            return

        if path == "/api/controls/reeflex/stop":
            payload = json.loads(body or "{}")
            self.proxy_control_post(("reeflex_stop", "arm_stop"), payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _vision_action(self, action, body: str) -> None:
        try:
            result = action(json.loads(body or "{}"))
            self.send_json({"status": "ok", "controller": result})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def send_json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def proxy_remote_snapshot(self, url: str) -> None:
        try:
            with urlrequest.urlopen(url, timeout=4) as response:
                data = response.read(512_000)
                content_type = response.headers.get("Content-Type", "image/jpeg")
        except (urlerror.URLError, TimeoutError, OSError):
            self.send_error(HTTPStatus.BAD_GATEWAY, "remote snapshot unavailable")
            return
        if not data:
            self.send_error(HTTPStatus.BAD_GATEWAY, "remote snapshot empty")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def proxy_control_get(self, keys: tuple[str, ...], node_id: Optional[str] = None, tank_id: Optional[str] = None) -> None:
        url = self.server.app.get_control_url(*keys, node_id=node_id, tank_id=tank_id)
        if not url:
            self.send_error(HTTPStatus.NOT_FOUND, "control URL unavailable")
            return
        try:
            with urlrequest.urlopen(url, timeout=3) as response:
                data = response.read(512_000)
                content_type = response.headers.get("Content-Type", "application/json")
                status = response.status
        except (urlerror.URLError, TimeoutError, OSError):
            self.send_error(HTTPStatus.BAD_GATEWAY, "control service unavailable")
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def proxy_control_post(self, keys: tuple[str, ...], payload: Dict) -> None:
        url = self.server.app.get_control_url(
            *keys,
            node_id=payload.get("node_id"),
            tank_id=payload.get("tank_id"),
        )
        if not url:
            self.send_error(HTTPStatus.NOT_FOUND, "control URL unavailable")
            return
        for key, value in payload.items():
            placeholder = "{" + key + "}"
            if placeholder in url:
                url = url.replace(placeholder, str(value))
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=3) as response:
                body = response.read(512_000)
                content_type = response.headers.get("Content-Type", "application/json")
                status = response.status
        except urlerror.HTTPError as exc:
            body = exc.read(512_000)
            content_type = exc.headers.get("Content-Type", "application/json")
            status = exc.code
        except (urlerror.URLError, TimeoutError, OSError):
            self.send_error(HTTPStatus.BAD_GATEWAY, "control service unavailable")
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def proxy_remote_stream(self, url: str) -> None:
        try:
            remote = urlrequest.urlopen(url, timeout=5)
        except (urlerror.URLError, TimeoutError, OSError):
            self.send_error(HTTPStatus.BAD_GATEWAY, "remote stream unavailable")
            return
        content_type = remote.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                chunk = remote.read(32_768)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            remote.close()

    def send_static(self, path: str) -> None:
        root = Path(__file__).resolve().parent / "static"
        relative = Path(path.lstrip("/")).relative_to("static")
        target = (root / relative).resolve()
        if root not in target.parents and target != root:
            self.send_error(HTTPStatus.FORBIDDEN, "forbidden")
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def index_html(self) -> str:
        return """<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>SEE SEA TV — Sync Tank</title>
    <link rel=\"stylesheet\" href=\"/static/app.css\">
  </head>
  <body>
    <main class=\"app-shell\">
      <header class=\"app-header\">
        <div class=\"brand-block\">
          <div class=\"app-eyebrow\">Habitat operations</div>
          <h1 class=\"brand-mark\"><span>Sync Tank</span></h1>
          <p id=\"system-guide\">Live habitat display</p>
        </div>
        <div class=\"status-strip\" id=\"feed-marquee\" aria-live=\"polite\"></div>
      </header>
      <section class=\"workspace\">
        <section class=\"primary-feed-stage\" id=\"primary-feed-stage\">
          <div class=\"cctv-bar\">
            <div><strong>SEE SEA TV</strong><span id=\"cctv-state\">Rotating every 8 seconds</span></div>
            <div class=\"cctv-actions\">
              <button id=\"feed-previous\">Previous</button><button id=\"feed-pin\">Pin</button>
              <button id=\"feed-next\">Next</button><button id=\"sighting-shutter\" class=\"shutter\">Capture</button>
              <button id=\"open-sightings\">Sightings</button>
            </div>
          </div>
          <div class=\"stage-feed-backdrop\" id=\"stage-feed-backdrop\" hidden>
            <img id=\"stage-feed-image\" alt=\"\">
            <div id=\"stage-feed-label\"></div>
          </div>
          <div class=\"feed-empty\" id=\"feed-empty\">No camera is connected. Simulator and Sightings remain available.</div>
          <div class=\"feed-thumbnails\" id=\"feed-thumbnails\"></div>
          <div class=\"primary-feed-hud\">
            <div>
              <div class=\"hud-title\" id=\"tank-title\">Sync Tank</div>
              <div class=\"hud-subtitle\" id=\"tank-subtitle\">Pick an unplaced feed, then click the tank.</div>
              <div class=\"tank-tabs\" id=\"tank-tabs\"></div>
            </div>
            <button class=\"setup-button icon-button\" id=\"setup-button\" aria-label=\"Open setup\">
              <span class=\"setup-glyph\" aria-hidden=\"true\"></span>
              <span class=\"sr-only\">Setup</span>
            </button>
          </div>
          <section class=\"lighthouse-control-section\" id=\"lighthouse-control-section\" hidden>
            <div class=\"lighthouse-header\">
              <div>
                <div class=\"section-title\">Raydar aim</div>
                <div class=\"lighthouse-readout\">
                  <span id=\"lighthouse-current\">Pan -- / Tilt --</span>
                  <span id=\"lighthouse-status\">Checking servos</span>
                </div>
              </div>
              <button class=\"lighthouse-close\" id=\"lighthouse-close\" aria-label=\"Close Raydar controls\">X</button>
            </div>
            <div class=\"lighthouse-pad\" aria-label=\"Raydar pan tilt controls\">
              <button data-lighthouse-move=\"up\" aria-label=\"Tilt up\">Up</button>
              <div>
                <button data-lighthouse-move=\"left\" aria-label=\"Pan left\">Left</button>
                <button data-lighthouse-move=\"center\" aria-label=\"Center pan tilt\">Center</button>
                <button data-lighthouse-move=\"right\" aria-label=\"Pan right\">Right</button>
              </div>
              <button data-lighthouse-move=\"down\" aria-label=\"Tilt down\">Down</button>
            </div>
            <div class=\"lighthouse-sliders\">
              <label>Pan <input id=\"lighthouse-pan-slider\" type=\"range\" min=\"20\" max=\"160\" step=\"1\"></label>
              <label>Tilt <input id=\"lighthouse-tilt-slider\" type=\"range\" min=\"45\" max=\"125\" step=\"1\"></label>
            </div>
            <div class=\"lighthouse-step\" role=\"group\" aria-label=\"Raydar step size\">
              <button data-lighthouse-step=\"1\">1 deg</button>
              <button data-lighthouse-step=\"3\" class=\"active\">3 deg</button>
              <button data-lighthouse-step=\"5\">5 deg</button>
            </div>
          </section>
          <section class=\"reeflex-control-section\" id=\"reeflex-control-section\" hidden>
            <div class=\"lighthouse-header\">
              <div>
                <div class=\"section-title\">Reeflex survey rig</div>
                <div class=\"lighthouse-readout\">
                  <span id=\"reeflex-current\">Base -- / Shoulder -- / Elbow --</span>
                  <span id=\"reeflex-status\">Checking servos</span>
                </div>
              </div>
              <button class=\"lighthouse-close\" id=\"reeflex-close\" aria-label=\"Close Reeflex controls\">X</button>
            </div>
            <div class=\"reeflex-joints\" aria-label=\"Reeflex axis controls\">
              <div class=\"reeflex-row\">
                <span>Base</span>
                <button data-reeflex-move=\"base-\" aria-label=\"Base left\">-</button>
                <input id=\"reeflex-base-slider\" type=\"range\" min=\"20\" max=\"160\" step=\"1\">
                <button data-reeflex-move=\"base+\" aria-label=\"Base right\">+</button>
              </div>
              <div class=\"reeflex-row\">
                <span>Shoulder</span>
                <button data-reeflex-move=\"shoulder-\" aria-label=\"Shoulder down\">-</button>
                <input id=\"reeflex-shoulder-slider\" type=\"range\" min=\"45\" max=\"135\" step=\"1\">
                <button data-reeflex-move=\"shoulder+\" aria-label=\"Shoulder up\">+</button>
              </div>
              <div class=\"reeflex-row\">
                <span>Elbow</span>
                <button data-reeflex-move=\"elbow-\" aria-label=\"Elbow down\">-</button>
                <input id=\"reeflex-elbow-slider\" type=\"range\" min=\"35\" max=\"145\" step=\"1\">
                <button data-reeflex-move=\"elbow+\" aria-label=\"Elbow up\">+</button>
              </div>
            </div>
            <div class=\"reeflex-actions\">
              <button data-reeflex-center>Center</button>
              <button data-reeflex-stop>Stop</button>
            </div>
            <div class=\"lighthouse-step\" role=\"group\" aria-label=\"Reeflex step size\">
              <button data-reeflex-step=\"1\">1 deg</button>
              <button data-reeflex-step=\"3\" class=\"active\">3 deg</button>
              <button data-reeflex-step=\"5\">5 deg</button>
            </div>
          </section>
        </section>
        <section class=\"tank-stage\" id=\"tank-stage\">
          <div class=\"simulator-heading\"><strong>Two-tank simulator</strong><span>Drag to orbit · wheel or pinch to zoom</span></div>
          <div class=\"tank-direction tank-direction-front\">FRONT</div><div class=\"tank-direction tank-direction-back\">BACK</div>
          <div class=\"tank-direction tank-direction-left\">LEFT</div><div class=\"tank-direction tank-direction-right\">RIGHT</div>
          <div class=\"tank-name tank-name-one\">TANK 1</div><div class=\"tank-name tank-name-two\">TANK 2</div>
          <div class=\"structure-toolbar\" id=\"structure-toolbar\">
            <select id=\"structure-type\"><option value=\"block\">Block</option><option value=\"slab\">Slab</option><option value=\"rounded-rock\">Rounded rock</option><option value=\"pillar\">Pillar</option><option value=\"arch\">Arch</option><option value=\"mound\">Mound</option></select>
            <button id=\"add-structure\">Add shape</button><button id=\"scatter-structures\">Scatter shapes</button>
          </div>
          <div class=\"autonomy-toolbar\"><span id=\"raydar-mode\">Raydar · STOP</span><button id=\"raydar-survey\">Survey</button><button id=\"raydar-stop\">STOP</button><span id=\"reeflex-mode\">Reeflex · STOP</span><button id=\"reeflex-survey\">Survey</button><button id=\"reeflex-auto-stop\">STOP</button></div>
          <div class=\"pip\" id=\"stage-pip\" hidden>
            <img id=\"pip-preview\" alt=\"\">
            <div id=\"pip-label\"></div>
          </div>
          <div class=\"world-controls\" id=\"world-controls\" hidden>
            <button data-world-move=\"up\">Up</button>
            <div>
              <button data-world-move=\"left\">Left</button>
              <button data-world-move=\"right\">Right</button>
            </div>
            <button data-world-move=\"down\">Down</button>
            <div>
              <button data-world-move=\"forward\">Forward</button>
              <button data-world-move=\"back\">Back</button>
            </div>
          </div>
        </section>
        <aside class=\"side-panel\" aria-label=\"Tank operations dock\">
          <section class=\"panel-section tank-summary-section\">
            <div class=\"section-title\">Tank status</div>
            <div class=\"tank-summary\" id=\"tank-summary\"></div>
          </section>
          <section class=\"panel-section placement-section\" id=\"placement-section\">
            <div class=\"section-title\">Place next</div>
            <div class=\"feed-list\" id=\"unplaced-list\"></div>
          </section>
          <section class=\"panel-section live-section\" id=\"live-section\" hidden>
            <div class=\"section-title\">SEE SEA TV views</div>
            <div class=\"live-viewer\">
              <img id=\"live-feed\" alt=\"\">
              <div class=\"live-meta\">
                <strong id=\"live-title\">Camera</strong>
                <span id=\"live-detail\">Rotating active feeds</span>
              </div>
            </div>
            <div class=\"live-controls\" id=\"live-controls\"></div>
          </section>
          <section class=\"panel-section observer-section\">
            <div class=\"section-title\">Motion focus</div>
            <div class=\"observation-list\" id=\"observation-list\"></div>
          </section>
          <section class=\"panel-section selected-only\" id=\"selected-section\" hidden>
            <div class=\"section-title\" id=\"selected-title\">Selected</div>
            <div class=\"preview\"><img id=\"feed-preview\" alt=\"\"></div>
            <div class=\"placement-pad\" id=\"placement-pad\" role=\"application\" aria-label=\"Drag placement control\">
              <div class=\"pad-label\">Drag to move</div>
              <div class=\"pad-cross\"></div>
              <div class=\"pad-stick\" id=\"placement-stick\"></div>
            </div>
            <div class=\"axis-row\" role=\"group\" aria-label=\"Placement mode\">
              <button data-axis=\"slide\" class=\"active\">Slide</button>
              <button data-axis=\"depth\">Depth</button>
              <button data-axis=\"aim\">Aim</button>
            </div>
            <div class=\"face-row\">
              <button data-face=\"x-\">Left wall</button>
              <button data-face=\"x+\">Right wall</button>
              <button data-face=\"z+\">Front</button>
              <button data-face=\"z-\">Back</button>
              <button data-face=\"y+\">Top</button>
            </div>
            <div class=\"control-grid\">
              <button data-move=\"left\">Left</button>
              <button data-move=\"right\">Right</button>
              <button data-move=\"up\">Up</button>
              <button data-move=\"down\">Down</button>
              <button data-move=\"forward\">Forward</button>
              <button data-move=\"back\">Back</button>
              <button data-move=\"rotate-left\">Rotate left</button>
              <button data-move=\"rotate-right\">Rotate right</button>
              <button data-move=\"aim-up\" class=\"endoscope-only\">Aim up</button>
              <button data-move=\"aim-down\" class=\"endoscope-only\">Aim down</button>
            </div>
            <div class=\"step-row\" role=\"group\" aria-label=\"Step size\">
              <button data-step=\"0.02\">Fine</button>
              <button data-step=\"0.06\" class=\"active\">Medium</button>
              <button data-step=\"0.14\">Large</button>
            </div>
            <div class=\"selected-actions\">
              <button id=\"identify-selected\">Identify feed</button>
              <button id=\"remove-selected\" class=\"danger-button\">Remove from tank</button>
            </div>
          </section>
        </aside>
        <button class=\"dock-toggle\" id=\"dock-toggle\" aria-label=\"Open placement controls\" aria-expanded=\"false\">
          <span class=\"dock-glyph\" aria-hidden=\"true\"></span>
          <span class=\"sr-only\">Placement controls</span>
        </button>
      </section>
    </main>
    <section class=\"sightings-drawer\" id=\"sightings-drawer\" hidden aria-label=\"Sightings album\">
      <header><div><strong>Sightings</strong><span>Wildlife moments saved locally</span></div><button id=\"close-sightings\">Close</button></header>
      <div class=\"sightings-grid\" id=\"sightings-grid\"></div>
    </section>
    <dialog class=\"deep-dialog\" id=\"deep-dialog\">
      <form method=\"dialog\"><h2>✦ Ask the Deep</h2><p>Sends this captured image to OpenAI for analysis</p>
        <img id=\"deep-image\" alt=\"The exact captured sighting that will be sent\">
        <div class=\"deep-actions\"><button value=\"cancel\">Cancel</button><button id=\"deep-confirm\" value=\"default\">Send this image</button></div>
      </form>
    </dialog>
    <section class=\"setup-overlay\" id=\"setup-overlay\" hidden>
      <div class=\"setup-card\">
        <div class=\"setup-kicker\">Tank setup</div>
        <h1 id=\"setup-question\">Choose tank size</h1>
        <div class=\"setup-current\" id=\"setup-current\"></div>
        <div class=\"setup-options\" id=\"setup-options\"></div>
        <div class=\"setup-actions\">
          <button id=\"setup-back\">Back</button>
          <button id=\"setup-skip\">Use defaults</button>
        </div>
      </div>
    </section>
    <script type=\"module\" src=\"/static/app.js\"></script>
  </body>
</html>"""


class TankManagerHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, app: TankManagerApp):
        super().__init__(server_address, handler_cls)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Sync Tank hub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage-dir", default=os.environ.get("SYNC_TANK_STORAGE_DIR", "./storage"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = TankManagerApp(storage_dir=args.storage_dir)
    server = TankManagerHTTPServer((args.host, args.port), TankManagerHandler, app)
    print(f"Tank Manager listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
