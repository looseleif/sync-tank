from __future__ import annotations

import json
import os
import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit

from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory

from sync_tank.cameras.usb import capture_usb_snapshot, list_video_devices, usb_camera_self_test, usb_mjpeg_command_candidates
from sync_tank.config import PROJECT_ROOT, load_config
from sync_tank.uplink import HubClient


NODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
PC_CAMERA_FIELDS = {
    "id",
    "camera_id",
    "label",
    "name",
    "camera_type",
    "source_type",
    "node_id",
    "relay_node_id",
    "tank_id",
    "latest_image_url",
    "snapshot_url",
    "stream_url",
    "preferred_live_url",
    "feed_mode",
    "content_type",
    "capture_command_url",
    "supports_capture_request",
    "status",
}

DEVICE_CATALOG = [
    {"device_type": "feeder", "label": "Feeder", "status": "available"},
    {"device_type": "floater", "label": "Floater", "status": "available"},
    {"device_type": "reeflex", "label": "REEFLEX", "status": "available"},
    {"device_type": "lighthouse", "label": "Lighthouse", "status": "available"},
    {"device_type": "scope", "label": "ReefScope", "status": "available"},
]


@dataclass
class IngestSettings:
    hub_id: str
    host_id: str
    host_label: str
    bind: str
    port: int
    upload_dir: Path
    state_path: Path
    layout_path: Path
    node_config_path: Path
    max_images_per_node: int
    expected_nodes: list[str]
    allowed_nodes: list[str]
    node_angles: dict[str, float]
    pc_sync_interval_seconds: int
    public_url: str
    camera_service_url: str
    usb_feed_allowed_cidrs: list[str]
    rig_profile: dict[str, Any]


class NodeStore:
    def __init__(self, settings: IngestSettings):
        self.settings = settings
        self._state_lock = threading.RLock()
        self._usb_cache_lock = threading.Lock()
        self._usb_cache_expires_at = 0.0
        self._usb_cache: list[dict[str, Any]] = []
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.settings.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.settings.state_path.exists():
            self._write_state({"nodes": {}})
        if not self.settings.layout_path.exists():
            self._write_layout(self._default_layout())
        if not self.settings.node_config_path.exists():
            self._write_node_config(self._default_node_config())
        self._reconcile_rig_profile()

    def _reconcile_rig_profile(self) -> None:
        """Apply a new checked-in rig profile without replacing local camera setup."""
        identity = self.settings.rig_profile or {}
        desired_profile = identity.get("profile") or {}
        profile_id = str(desired_profile.get("id") or "")
        if not profile_id:
            return

        config = self.node_config()
        tank_id = str((identity.get("tank") or {}).get("id") or "")
        node = identity.get("node") or {}
        config["node"] = {
            **dict(config.get("node") or {}),
            "id": str(node.get("id") or self.settings.host_id),
            "label": str(node.get("label") or self.settings.host_label),
        }
        inventory = identity.get("inventory") or {}
        node_inventory = config.setdefault("inventory", {})
        node_inventory["lighthouses"] = int(inventory.get("lighthouse_cameras", 0))
        node_inventory["robotic_arms"] = int(inventory.get("reeflex_arms", 0))
        config["profile"] = desired_profile
        role_split = desired_profile.get("role_split") or {}
        cameras = config.get("cameras") or {}
        for camera in cameras.values():
            camera["node_id"] = self.settings.host_id
            if tank_id:
                camera["tank_id"] = tank_id
        if role_split.get("lighthouse") and not role_split.get("reeflex"):
            for camera in cameras.values():
                if camera.get("camera_type") == "reeflex_cam":
                    camera["camera_type"] = "lighthouse_cam"
                    if "reeflex" in str(camera.get("label", "")).lower():
                        camera["label"] = "Raydar Camera #1"
        elif role_split.get("reeflex") and not role_split.get("lighthouse"):
            for camera in cameras.values():
                if camera.get("camera_type") == "lighthouse_cam":
                    camera["camera_type"] = "reeflex_cam"
                    if any(name in str(camera.get("label", "")).lower() for name in ("lighthouse", "raydar")):
                        camera["label"] = "Reeflex Camera #1"
        self._write_node_config(config)

    def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        node_id = _clean_node_id(str(payload.get("node_id", "")))
        hub_id = str(payload.get("hub_id") or payload.get("host_id") or "")
        self._validate_node_assignment(node_id, hub_id, require_hub_id=True)
        node = self._node(node_id)
        node.update(
            {
                "node_id": node_id,
                "node_type": payload.get("node_type"),
                "hub_id": hub_id or self.settings.hub_id,
                "firmware": payload.get("firmware"),
                "uptime_ms": payload.get("uptime_ms"),
                "wifi_rssi": payload.get("wifi_rssi"),
                "free_heap": payload.get("free_heap"),
                "battery_mv": payload.get("battery_mv"),
                "camera_available": payload.get("camera_available"),
                "last_image_upload_status": payload.get("last_image_upload_status"),
                "status": payload.get("status", "online"),
                "host_id": str(payload.get("host_id") or payload.get("hub_id") or self.settings.host_id),
                "receiver_host_id": self.settings.host_id,
                "last_heartbeat_at": _now(),
            }
        )
        self._put_node(node_id, node)
        return node

    def image_upload(self, node_id: str, image: bytes, headers: dict[str, str]) -> dict[str, Any]:
        node_id = _clean_node_id(node_id)
        hub_id = headers.get("X-Hub-Id") or headers.get("X-Host-Id") or ""
        self._validate_node_assignment(node_id, hub_id, require_hub_id=True)
        if not image.startswith(b"\xff\xd8"):
            raise ValueError("Body does not look like a JPEG")

        node_dir = self.settings.upload_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        filename = f"{node_id}_{timestamp}.jpg"
        image_path = node_dir / filename
        image_path.write_bytes(image)

        node = self._node(node_id)
        node.update(
            {
                "node_id": node_id,
                "node_type": headers.get("X-Node-Type"),
                "hub_id": hub_id or self.settings.hub_id,
                "firmware": headers.get("X-Firmware-Version"),
                "uptime_ms": _int_or_none(headers.get("X-Uptime-Ms")),
                "wifi_rssi": _int_or_none(headers.get("X-Wifi-Rssi")),
                "free_heap": _int_or_none(headers.get("X-Free-Heap")),
                "image_format": headers.get("X-Image-Format", "jpeg"),
                "image_size_bytes": len(image),
                "capture_timestamp_ms": _int_or_none(headers.get("X-Capture-Timestamp-Ms")),
                "status": "online",
                "host_id": headers.get("X-Host-Id") or headers.get("X-Hub-Id") or self.settings.host_id,
                "receiver_host_id": self.settings.host_id,
                "last_image_at": _now(),
                "latest_image": {
                    "filename": filename,
                    "url": f"/uploads/{node_id}/{filename}",
                    "size_bytes": len(image),
                },
            }
        )
        self._put_node(node_id, node)
        self._record_stream_frame(node_id, filename, len(image))
        self._rotate(node_id)
        return node

    def set_node_command(self, node_id: str, command: dict[str, Any]) -> dict[str, Any]:
        node_id = _clean_node_id(node_id)
        self._validate_node_assignment(node_id, self.settings.hub_id)
        normalized = self._normalize_command(command)
        state = self._read_state()
        state.setdefault("pending_commands", {})[node_id] = {
            **normalized,
            "queued_at": _now(),
            "hub_id": self.settings.hub_id,
        }
        if normalized.get("command") == "stream":
            state.setdefault("active_stream_sessions", {}).pop(node_id, None)
        self._write_state(state)
        return state["pending_commands"][node_id]

    def pop_node_command(self, node_id: str) -> dict[str, Any] | None:
        node_id = _clean_node_id(node_id)
        self._validate_node_assignment(node_id, self.settings.hub_id)
        state = self._read_state()
        pending = state.setdefault("pending_commands", {})
        command = pending.pop(node_id, None)
        if command is not None:
            delivered_at = _now()
            if command.get("command") == "stream":
                self._start_stream_session(state, node_id, command, delivered_at)
            state.setdefault("command_history", []).append(
                {
                    "node_id": node_id,
                    "hub_id": self.settings.hub_id,
                    "command": command,
                    "delivered_at": delivered_at,
                }
            )
            state["command_history"] = state["command_history"][-50:]
            self._write_state(state)
        return command

    def pending_commands(self) -> dict[str, Any]:
        return self._read_state().get("pending_commands", {})

    def stream_session(self, node_id: str) -> dict[str, Any] | None:
        node_id = _clean_node_id(node_id)
        state = self._read_state()
        session = state.get("active_stream_sessions", {}).get(node_id)
        if not session:
            return None
        changed = self._refresh_stream_session_state(session)
        if changed:
            state.setdefault("active_stream_sessions", {})[node_id] = session
            self._write_state(state)
        return session

    def stream_history(self, node_id: str, limit: int = 5) -> list[dict[str, Any]]:
        node_id = _clean_node_id(node_id)
        state = self._read_state()
        sessions = [
            session
            for session in state.get("stream_sessions", {}).values()
            if session.get("node_id") == node_id
        ]
        active = state.get("active_stream_sessions", {}).get(node_id)
        seen = {session.get("session_id") for session in sessions}
        if active and active.get("session_id") not in seen:
            sessions.append(active)
        changed = False
        for session in sessions:
            if not session.get("frames"):
                frames = self._session_frames_from_disk(node_id, session)
                if frames:
                    session["frames"] = frames
                    changed = True
            if self._refresh_stream_session_state(session):
                changed = True
        if changed:
            for session in sessions:
                session_id = session.get("session_id")
                if session_id:
                    state.setdefault("stream_sessions", {})[session_id] = session
            self._write_state(state)

        def session_sort_key(session: dict[str, Any]) -> str:
            return str(session.get("command_returned_at") or session.get("requested_at") or "")

        return sorted(sessions, key=session_sort_key, reverse=True)[:limit]

    def latest_stream_frame(self, node_id: str) -> tuple[Path, dict[str, Any]]:
        session = self.stream_session(node_id)
        if not session or not session.get("latest_frame"):
            raise FileNotFoundError("No active stream frame for node")
        frame = session["latest_frame"]
        path = self.settings.upload_dir / node_id / frame["filename"]
        if not path.exists():
            raise FileNotFoundError("Latest stream frame file is missing")
        return path, session

    def summary(self) -> dict[str, Any]:
        state = self._read_state()
        nodes = state.get("nodes", {})
        for node_id in self.settings.expected_nodes:
            nodes.setdefault(node_id, {"node_id": node_id, "status": "waiting"})
        dashboard_url = self.settings.public_url or f"http://{self.settings.bind}:{self.settings.port}"
        camera_service_url = self.settings.camera_service_url or dashboard_url.replace(f":{self.settings.port}", ":5050")
        return {
            "nodes": nodes,
            "expected_nodes": self.settings.expected_nodes,
            "node_angles": self.settings.node_angles,
            "pending_commands": self.pending_commands(),
            "stream_sessions": state.get("active_stream_sessions", {}),
            "layout": self.layout(),
            "node_config": self.node_config(),
            "inventory": self.inventory(),
            "device_inventory": self.device_inventory(),
            "device_catalog": self.device_catalog(),
            "setup_state": self.setup_state(),
            "consolidated_cameras": self.consolidated_cameras(nodes),
            "systems": self.systems_status(nodes),
            "host": {"id": self.settings.host_id, "hub_id": self.settings.hub_id, "label": self.settings.host_label},
            "usb_cameras": self.usb_cameras(),
            "upload_dir": str(self.settings.upload_dir),
            "boot": {
                "project_root": str(PROJECT_ROOT),
                "dashboard_url": dashboard_url,
                "camera_service_url": camera_service_url,
                "bind": self.settings.bind,
                "port": self.settings.port,
                "upload_dir": str(self.settings.upload_dir),
                "state_path": str(self.settings.state_path),
                "node_config_path": str(self.settings.node_config_path),
                "commands": {
                    "start_dashboard": f"cd {PROJECT_ROOT} && ./scripts/run-ingest.sh",
                    "start_camera_service": f"cd {PROJECT_ROOT} && ./scripts/run-dev.sh",
                    "self_test": f"cd {PROJECT_ROOT} && ./scripts/self-test-cameras.sh --repair",
                    "register_pc_hub": f"cd {PROJECT_ROOT} && ./scripts/register-pc-hub.sh http://PC_IP:8765",
                },
            },
        }

    def node_config(self) -> dict[str, Any]:
        with self.settings.node_config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        changed = self._ensure_node_config_devices(config)
        if changed:
            self._write_node_config(config)
        return config

    def save_node_config(self, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config.get("inventory"), dict):
            raise ValueError("Node config must include inventory")
        if not isinstance(config.get("cameras"), dict):
            config["cameras"] = {}
        if not isinstance(config.get("feeders"), dict):
            config["feeders"] = {"solid": 0, "liquid": 0, "misc": 0}
        if not isinstance(config.get("feeder_viewports"), dict):
            config["feeder_viewports"] = {}
        if not isinstance(config.get("validation"), dict):
            config["validation"] = {"status": "default", "validated_by_hand": False}
        config.setdefault("node", {"id": self.settings.host_id, "label": self.settings.host_label})
        self._write_node_config(config)
        return self.node_config()

    def inventory(self) -> dict[str, Any]:
        config = self.node_config()
        inventory = config.get("inventory", {})
        feeders = config.get("feeders", {})
        return {
            "robotic_arms": int(inventory.get("robotic_arms", 1)),
            "endoscope_cameras": int(inventory.get("endoscope_cameras", len(self.usb_cameras()))),
            "floater_cameras": int(inventory.get("floater_cameras", len(self.settings.expected_nodes))),
            "lighthouses": int(inventory.get("lighthouses", 0)),
            "feeders": {
                "solid": int(feeders.get("solid", 0)),
                "liquid": int(feeders.get("liquid", 0)),
                "misc": int(feeders.get("misc", 0)),
            },
        }

    def device_inventory(self) -> dict[str, Any]:
        inventory = self.inventory()
        feeder_count = sum(inventory["feeders"].values())
        counts = {
            "feeder": feeder_count,
            "floater": inventory["floater_cameras"],
            "reeflex": inventory["robotic_arms"],
            "lighthouse": inventory["lighthouses"],
            "scope": inventory["endoscope_cameras"],
        }
        return {
            "active_tank_nodes": 1,
            "counts": counts,
            "owned_device_count": sum(counts.values()),
            "device_types": [item["device_type"] for item in DEVICE_CATALOG],
        }

    def device_catalog(self) -> list[dict[str, str]]:
        return [dict(item) for item in DEVICE_CATALOG]

    def setup_state(self) -> dict[str, Any]:
        config = self.node_config()
        validation = config.get("validation") or {}
        status = str(validation.get("status") or "default")
        validated_by_hand = bool(validation.get("validated_by_hand") or status == "validated")
        return {
            "status": "validated" if validated_by_hand else "default",
            "validated_by_hand": validated_by_hand,
            "validated_at": validation.get("validated_at"),
            "validated_by": validation.get("validated_by"),
            "message": "device inventory validated by hand" if validated_by_hand else "default setup; device inventory needs hand validation",
        }

    def systems_status(self, nodes: dict[str, Any] | None = None) -> dict[str, Any]:
        nodes = nodes or self._read_state().get("nodes", {})
        inventory = self.inventory()
        usb = self.usb_cameras()
        config = self.node_config()
        camera_config = config.get("cameras", {})
        detected_usb_ids = {camera["id"] for camera in usb}
        detected_reefscope_ids = [
            camera_id
            for camera_id in detected_usb_ids
            if (camera_config.get(camera_id, {}).get("camera_type") or "endoscope_cam") == "endoscope_cam"
        ]
        detected_lighthouse_ids = [
            camera_id
            for camera_id in detected_usb_ids
            if camera_config.get(camera_id, {}).get("camera_type") == "lighthouse_cam"
        ]
        floaters = [nodes.get(node_id, {"node_id": node_id, "status": "waiting"}) for node_id in self.settings.expected_nodes]
        active_floaters = [node for node in floaters if self._recent(node.get("last_image_at"), stale_seconds=90)]
        feeders = inventory["feeders"]
        feeder_total = sum(feeders.values())
        lighthouses = int(inventory.get("lighthouses", 0))
        issues = []
        if len(detected_reefscope_ids) < inventory["endoscope_cameras"]:
            issues.append(f"Expected {inventory['endoscope_cameras']} ReefScope cameras, detected {len(detected_reefscope_ids)}.")
        if len(detected_lighthouse_ids) < lighthouses:
            issues.append(f"Expected {lighthouses} Lighthouse camera, detected {len(detected_lighthouse_ids)}.")
        if len(active_floaters) < len(self.settings.expected_nodes):
            issues.append(f"{len(active_floaters)} of {len(self.settings.expected_nodes)} Floater cameras uploaded recently.")
        if feeder_total and not config.get("feeder_viewports"):
            issues.append("Feeders are configured but no feeder viewports are assigned.")
        return {
            "reefscope": {"expected": inventory["endoscope_cameras"], "active": len(detected_reefscope_ids), "status": "ok" if len(detected_reefscope_ids) >= inventory["endoscope_cameras"] else "warn"},
            "floaters": {"expected": len(self.settings.expected_nodes), "active": len(active_floaters), "status": "ok" if len(active_floaters) == len(self.settings.expected_nodes) else "warn"},
            "reeflex": {"expected": inventory["robotic_arms"], "active": inventory["robotic_arms"], "status": "ok" if inventory["robotic_arms"] else "idle"},
            "lighthouse": {"expected": lighthouses, "active": len(detected_lighthouse_ids), "status": "ok" if len(detected_lighthouse_ids) >= lighthouses else ("warn" if lighthouses else "idle")},
            "feeders": {"expected": feeder_total, "active": feeder_total, "status": "ok" if feeder_total else "idle", "by_type": feeders},
            "issues": issues,
        }

    def consolidated_cameras(self, nodes: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        nodes = nodes or self._read_state().get("nodes", {})
        config = self.node_config()
        camera_config = config.get("cameras", {})
        cameras = []
        for node_id in self.settings.expected_nodes:
            node = nodes.get(node_id, {"node_id": node_id, "status": "waiting"})
            configured = camera_config.get(node_id, {})
            latest = node.get("latest_image") or {}
            status = self._esp32_camera_status(node)
            cameras.append(
                {
                    "camera_id": node_id,
                    "label": configured.get("label", node_id),
                    "camera_type": configured.get("camera_type", "floater_cam"),
                    "source_type": "esp32_upload",
                    "node_id": configured.get("node_id", self.settings.host_id),
                    "relay_node_id": configured.get("relay_node_id"),
                    "hub_id": self.settings.hub_id,
                    "tank_id": configured.get("tank_id"),
                    "status": status,
                    "latest_image_url": latest.get("url"),
                    "last_image_at": node.get("last_image_at"),
                    "wifi_rssi": node.get("wifi_rssi"),
                    "enabled": configured.get("enabled", True),
                }
            )
        for camera in self.usb_cameras():
            configured = camera_config.get(camera["id"], {})
            cameras.append(
                {
                    "camera_id": camera["id"],
                    "label": configured.get("label", camera["name"]),
                    "camera_type": configured.get("camera_type", "endoscope_cam"),
                    "source_type": "usb_camera",
                    "node_id": configured.get("node_id", self.settings.host_id),
                    "relay_node_id": configured.get("relay_node_id"),
                    "hub_id": self.settings.hub_id,
                    "tank_id": configured.get("tank_id"),
                    "status": camera.get("status", "online"),
                    "snapshot_url": f"/api/usb/{camera['id']}/snapshot",
                    "stable_match": configured.get("stable_match") or camera.get("stable_match", {}),
                    "enabled": configured.get("enabled", True),
                }
            )
        existing_ids = {camera["camera_id"] for camera in cameras}
        for camera_id, configured in camera_config.items():
            if camera_id in existing_ids or not configured.get("enabled", True):
                continue
            source_type = configured.get("source_type") or self._configured_source_type(camera_id, configured)
            if source_type == "usb_camera":
                cameras.append(
                    {
                        "camera_id": camera_id,
                        "label": configured.get("label", camera_id),
                        "camera_type": configured.get("camera_type", "endoscope_cam"),
                        "source_type": "usb_camera",
                        "node_id": configured.get("node_id", self.settings.host_id),
                        "relay_node_id": configured.get("relay_node_id"),
                        "hub_id": self.settings.hub_id,
                        "tank_id": configured.get("tank_id"),
                        "status": "offline",
                        "snapshot_url": f"/api/usb/{camera_id}/snapshot",
                        "stable_match": configured.get("stable_match", {}),
                        "enabled": configured.get("enabled", True),
                    }
                )
                continue
            node = nodes.get(camera_id, {"node_id": camera_id, "status": "waiting"})
            latest = node.get("latest_image") or {}
            status = self._esp32_camera_status(node)
            cameras.append(
                {
                    "camera_id": camera_id,
                    "label": configured.get("label", camera_id),
                    "camera_type": configured.get("camera_type", "floater_cam"),
                    "source_type": source_type,
                    "node_id": configured.get("node_id", self.settings.host_id),
                    "relay_node_id": configured.get("relay_node_id"),
                    "hub_id": self.settings.hub_id,
                    "tank_id": configured.get("tank_id"),
                    "status": status,
                    "latest_image_url": latest.get("url"),
                    "last_image_at": node.get("last_image_at"),
                    "wifi_rssi": node.get("wifi_rssi"),
                    "enabled": configured.get("enabled", True),
                }
            )
        return cameras

    def hub_payload(self) -> dict[str, Any]:
        summary = self.summary()
        return {
            "node": {
                "node_id": self.settings.host_id,
                "hub_id": self.settings.hub_id,
                "node_type": "raspberry_pi_tank_node",
                "label": self.settings.host_label,
                "status": "online",
                "last_seen": _now(),
            },
            "inventory": summary["inventory"],
            "device_inventory": summary["device_inventory"],
            "device_catalog": summary["device_catalog"],
            "setup_state": summary["setup_state"],
            "cameras": summary["consolidated_cameras"],
            "layout": summary["layout"],
        }

    def pc_node_payload(self, lan_url: str) -> dict[str, Any]:
        return {
            "node_id": self.settings.host_id,
            "node_type": "raspberry_pi_tank_node",
            "hostname": socket.gethostname(),
            "label": self.settings.host_label,
            "tank_ids": self._tank_ids(),
            "lan_url": lan_url.rstrip("/"),
            "status": "online",
            "active_tank_node_count": 1,
            "setup_state": self.setup_state(),
            "device_inventory": self.device_inventory(),
            "device_catalog": self.device_catalog(),
            "control_urls": self.control_urls(lan_url),
        }

    def control_urls(self, lan_url: str) -> dict[str, str]:
        control_base_url = _reachable_service_url(self.settings.camera_service_url, lan_url, 5050)
        urls = {
            "arm_status": f"{control_base_url}/api/arm",
            "servo_channel": f"{control_base_url}/api/servo/channel",
        }
        counts = self.device_inventory()["counts"]
        if counts.get("reeflex", 0):
            urls.update({
                "reeflex_status": f"{control_base_url}/api/arm",
                "reeflex_stop": f"{control_base_url}/api/arm/stop",
                "reeflex_pose": f"{control_base_url}/api/reeflex/pose",
                "reeflex_servo": f"{control_base_url}/api/arm/servo/{{servo_id}}",
                "reeflex_idle": f"{control_base_url}/api/reeflex/idle",
                "reeflex_idle_start": f"{control_base_url}/api/reeflex/idle/start",
                "reeflex_idle_stop": f"{control_base_url}/api/reeflex/idle/stop",
            })
        if counts.get("lighthouse", 0):
            urls["lighthouse_pose"] = f"{control_base_url}/api/lighthouse/pose"
            urls["lighthouse_survey"] = f"{control_base_url}/api/lighthouse/survey"
            urls["lighthouse_survey_start"] = f"{control_base_url}/api/lighthouse/survey/start"
            urls["lighthouse_survey_stop"] = f"{control_base_url}/api/lighthouse/survey/stop"
        return urls

    def pc_camera_inventory(self, lan_url: str, include_relayed: bool = False) -> list[dict[str, Any]]:
        base_url = lan_url.rstrip("/")
        camera_base_url = (self.settings.camera_service_url or base_url).rstrip("/")
        cameras = []
        for camera in self.consolidated_cameras():
            is_relayed = bool(camera.get("relay_node_id")) or camera.get("node_id") != self.settings.host_id
            if is_relayed != include_relayed:
                continue
            item = {
                "id": camera["camera_id"],
                "camera_id": camera["camera_id"],
                "label": camera.get("label") or camera["camera_id"],
                "name": camera.get("label") or camera["camera_id"],
                "camera_type": camera.get("camera_type"),
                "source_type": camera.get("source_type"),
                "node_id": camera.get("node_id") or self.settings.host_id,
                "relay_node_id": camera.get("relay_node_id"),
                "tank_id": camera.get("tank_id"),
                "status": camera.get("status", "offline"),
            }
            if camera.get("source_type") == "esp32_upload":
                item["latest_image_url"] = f"{base_url}/uploads/{camera['camera_id']}/latest.jpg"
                if camera.get("node_id") == self.settings.host_id and not camera.get("relay_node_id"):
                    item["capture_command_url"] = f"{base_url}/api/node/{camera['camera_id']}/capture"
                    item["supports_capture_request"] = True
            if camera.get("source_type") == "usb_camera":
                item["snapshot_url"] = f"{base_url}/api/usb/{camera['camera_id']}/snapshot"
                item["stream_url"] = f"{base_url}/api/usb/{camera['camera_id']}/stream"
                item["preferred_live_url"] = item["stream_url"]
                item["feed_mode"] = "mjpeg"
                item["content_type"] = "multipart/x-mixed-replace; boundary=frame"
            cameras.append({key: value for key, value in item.items() if key in PC_CAMERA_FIELDS and value not in (None, "")})
        return cameras

    def pc_contract_payload(self, lan_url: str) -> dict[str, Any]:
        return {
            "node": self.pc_node_payload(lan_url),
            "camera_registration": {
                "node_id": self.settings.host_id,
                "scope": "owned_by_node",
                "cameras": self.pc_camera_inventory(lan_url, include_relayed=False),
            },
            "relayed_camera_registration": {
                "node_id": self.settings.host_id,
                "scope": "relayed_by_node",
                "cameras": self.pc_camera_inventory(lan_url, include_relayed=True),
            },
            "heartbeat": {"node_id": self.settings.host_id, "status": "online"},
        }

    def layout(self) -> dict[str, Any]:
        with self.settings.layout_path.open("r", encoding="utf-8") as handle:
            layout = json.load(handle)
        self._ensure_layout_devices(layout)
        return layout

    def save_layout(self, layout: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(layout.get("tanks"), list) or not layout["tanks"]:
            raise ValueError("Layout must include at least one tank")
        if not isinstance(layout.get("hosts"), list):
            layout["hosts"] = []
        if not isinstance(layout.get("cameras"), list):
            layout["cameras"] = []
        layout.setdefault("active_tank_id", layout["tanks"][0]["id"])
        layout.setdefault("show_internals", True)
        self._write_layout(layout)
        return self.layout()

    def usb_cameras(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._usb_cache_lock:
            if now < self._usb_cache_expires_at:
                return [self._copy_usb_camera(camera) for camera in self._usb_cache]

        aliases = self._configured_usb_aliases()
        cameras = []
        for camera in list_video_devices():
            stable_match = camera.get("stable_match", {})
            camera_id = aliases.get(stable_match.get("by_path")) or aliases.get(stable_match.get("id_path")) or camera["id"]
            cameras.append(
                {
                    "id": camera_id,
                    "detected_id": camera["id"],
                    "name": camera.get("name", camera["id"]),
                    "source_type": "usb",
                    "device": camera.get("device", ""),
                    "host_id": self.settings.host_id,
                    "hub_id": self.settings.hub_id,
                    "stable_match": stable_match,
                    "status": camera.get("status", "online"),
                }
            )
        with self._usb_cache_lock:
            self._usb_cache = [self._copy_usb_camera(camera) for camera in cameras]
            self._usb_cache_expires_at = time.monotonic() + 2.0
        return cameras

    def _copy_usb_camera(self, camera: dict[str, Any]) -> dict[str, Any]:
        copied = dict(camera)
        copied["stable_match"] = dict(camera.get("stable_match", {}))
        return copied

    def _configured_usb_aliases(self) -> dict[str, str]:
        if not self.settings.node_config_path.exists():
            return {}
        try:
            with self.settings.node_config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception:
            return {}
        aliases = {}
        for camera_id, configured in (config.get("cameras") or {}).items():
            if configured.get("camera_type") != "endoscope_cam" and configured.get("source_type") != "usb_camera":
                continue
            stable_match = configured.get("stable_match") or {}
            by_path = stable_match.get("by_path")
            id_path = stable_match.get("id_path")
            if by_path and str(by_path) not in aliases:
                aliases[str(by_path)] = str(camera_id)
            if id_path and str(id_path) not in aliases:
                aliases[str(id_path)] = str(camera_id)
        return aliases

    def _configured_usb_id_for_match(self, config: dict[str, Any], stable_match: dict[str, Any]) -> str:
        by_path = stable_match.get("by_path")
        id_path = stable_match.get("id_path")
        for camera_id, configured in (config.get("cameras") or {}).items():
            if configured.get("camera_type") != "endoscope_cam" and configured.get("source_type") != "usb_camera":
                continue
            configured_match = configured.get("stable_match") or {}
            if by_path and configured_match.get("by_path") == by_path:
                return str(camera_id)
            if id_path and configured_match.get("id_path") == id_path:
                return str(camera_id)
        return ""

    def _recent(self, value: Any, stale_seconds: int) -> bool:
        if not value:
            return False
        try:
            return datetime.now(timezone.utc) - _parse_iso(str(value)) <= timedelta(seconds=stale_seconds)
        except ValueError:
            return False

    def _esp32_camera_status(self, node: dict[str, Any]) -> str:
        if self._recent(node.get("last_image_at"), stale_seconds=90):
            return str(node.get("status") or "online")
        if node.get("last_image_at"):
            return "stale"
        return str(node.get("status") or "waiting")

    def _node(self, node_id: str) -> dict[str, Any]:
        return self._read_state().get("nodes", {}).get(node_id, {"node_id": node_id})

    def _put_node(self, node_id: str, node: dict[str, Any]) -> None:
        state = self._read_state()
        state.setdefault("nodes", {})[node_id] = node
        self._write_state(state)

    def _read_state(self) -> dict[str, Any]:
        with self._state_lock:
            with self.settings.state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    def _write_state(self, state: dict[str, Any]) -> None:
        with self._state_lock:
            self.settings.state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.settings.state_path.with_name(f".{self.settings.state_path.name}.tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.flush()
            temp_path.replace(self.settings.state_path)

    def _write_layout(self, layout: dict[str, Any]) -> None:
        self.settings.layout_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.layout_path.open("w", encoding="utf-8") as handle:
            json.dump(layout, handle, indent=2, sort_keys=True)

    def _write_node_config(self, config: dict[str, Any]) -> None:
        self.settings.node_config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.node_config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    def _default_node_config(self) -> dict[str, Any]:
        cameras = {
            node_id: {
                "label": f"Floater Camera #{index}",
                "camera_type": "floater_cam",
                "node_id": self.settings.host_id,
                "tank_id": "tank-main",
                "enabled": True,
            }
            for index, node_id in enumerate(self.settings.expected_nodes, start=1)
        }
        for index, camera in enumerate(self.usb_cameras(), start=1):
            cameras[camera["id"]] = {
                "label": f"ReefScope Camera #{index}",
                "camera_type": "endoscope_cam",
                "source_type": "usb_camera",
                "node_id": self.settings.host_id,
                "tank_id": "tank-main",
                "stable_match": camera.get("stable_match", {}),
                "enabled": True,
            }
        return {
            "node": {"id": self.settings.host_id, "label": self.settings.host_label, "role": "edge_capture_node"},
            "inventory": {
                "robotic_arms": 1,
                "endoscope_cameras": len(self.usb_cameras()),
                "floater_cameras": len(self.settings.expected_nodes),
                "lighthouses": 0,
            },
            "validation": {"status": "default", "validated_by_hand": False},
            "feeders": {"solid": 0, "liquid": 0, "misc": 0},
            "feeder_viewports": {},
            "cameras": cameras,
            "notes": "",
        }

    def _ensure_node_config_devices(self, config: dict[str, Any]) -> bool:
        changed = False
        config.setdefault("node", {"id": self.settings.host_id, "label": self.settings.host_label, "role": "edge_capture_node"})
        config.setdefault("inventory", {})
        config["inventory"].setdefault("lighthouses", 0)
        config.setdefault("validation", {"status": "default", "validated_by_hand": False})
        config.setdefault("feeders", {"solid": 0, "liquid": 0, "misc": 0})
        config.setdefault("feeder_viewports", {})
        config.setdefault("cameras", {})
        for index, node_id in enumerate(self.settings.expected_nodes, start=1):
            if node_id not in config["cameras"]:
                config["cameras"][node_id] = {
                    "label": f"Floater Camera #{index}",
                    "camera_type": "floater_cam",
                    "node_id": self.settings.host_id,
                    "tank_id": "tank-main",
                    "enabled": True,
                }
                changed = True
            configured = config["cameras"][node_id]
            if not configured.get("node_id"):
                configured["node_id"] = self.settings.host_id
                changed = True
            if not configured.get("tank_id"):
                configured["tank_id"] = "tank-main"
                changed = True
            if not configured.get("source_type"):
                configured["source_type"] = "esp32_upload"
                changed = True
        for index, camera in enumerate(self.usb_cameras(), start=1):
            configured_for_device = self._configured_usb_id_for_match(config, camera.get("stable_match", {}))
            if configured_for_device and configured_for_device != camera["id"]:
                continue
            if camera["id"] not in config["cameras"]:
                config["cameras"][camera["id"]] = {
                    "label": f"ReefScope Camera #{index}",
                    "camera_type": "endoscope_cam",
                    "source_type": "usb_camera",
                    "node_id": self.settings.host_id,
                    "tank_id": "tank-main",
                    "stable_match": camera.get("stable_match", {}),
                    "enabled": True,
                }
                changed = True
            configured = config["cameras"][camera["id"]]
            if not configured.get("node_id"):
                configured["node_id"] = self.settings.host_id
                changed = True
            if not configured.get("tank_id"):
                configured["tank_id"] = "tank-main"
                changed = True
            if not configured.get("source_type") or configured.get("source_type") != "usb_camera":
                configured["source_type"] = "usb_camera"
                changed = True
            if camera.get("stable_match") and configured.get("stable_match") != camera.get("stable_match"):
                configured["stable_match"] = camera.get("stable_match", {})
                changed = True
        return changed

    def _configured_source_type(self, camera_id: str, configured: dict[str, Any]) -> str:
        if configured.get("source_type"):
            return str(configured["source_type"])
        if configured.get("camera_type") == "endoscope_cam" or camera_id.startswith("usb_"):
            return "usb_camera"
        return "esp32_upload"

    def _default_layout(self) -> dict[str, Any]:
        cameras = []
        for index, node_id in enumerate(self.settings.expected_nodes):
            cameras.append(
                {
                    "id": node_id,
                    "label": f"Floater Camera #{index + 1}",
                    "source_type": "esp32",
                    "host_id": self.settings.host_id,
                    "tank_id": "tank-main",
                    "angle": self.settings.node_angles.get(node_id, index * 90),
                    "radius": 1.08,
                    "position": _position_from_angle(self.settings.node_angles.get(node_id, index * 90), 1.08),
                    "target": {"x": 0, "y": 0, "z": 0},
                    "fov": 56,
                    "enabled": True,
                }
            )
        return {
            "active_tank_id": "tank-main",
            "show_internals": True,
            "tanks": [
                {
                    "id": "tank-main",
                    "label": "Main Tank",
                    "style": "cube",
                    "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0},
                    "internals": [
                        {"id": "hide-1", "label": "Hide", "x": -0.22, "y": 0.1, "z": 0.05},
                        {"id": "plant-1", "label": "Planting", "x": 0.28, "y": -0.08, "z": -0.1},
                    ],
                }
            ],
            "hosts": [
                {
                    "id": self.settings.host_id,
                    "label": self.settings.host_label,
                    "tank_id": "tank-main",
                    "x": 0,
                    "y": 0.72,
                    "position": {"x": 0, "y": -0.75, "z": 1.2},
                }
            ],
            "cameras": cameras,
        }

    def _ensure_layout_devices(self, layout: dict[str, Any]) -> None:
        changed = self._ensure_3d_layout_fields(layout)
        existing = {camera.get("id") for camera in layout.get("cameras", [])}
        for index, node_id in enumerate(self.settings.expected_nodes):
            if node_id in existing:
                continue
            angle = self.settings.node_angles.get(node_id, index * 90)
            layout.setdefault("cameras", []).append(
                {
                    "id": node_id,
                    "label": f"Floater Camera #{index + 1}",
                    "source_type": "esp32",
                    "host_id": self.settings.host_id,
                    "tank_id": layout.get("active_tank_id", "tank-main"),
                    "angle": angle,
                    "radius": 1.08,
                    "position": _position_from_angle(angle, 1.08),
                    "target": {"x": 0, "y": 0, "z": 0},
                    "fov": 56,
                    "enabled": True,
                }
            )
            existing.add(node_id)
            changed = True
        for index, camera in enumerate(self.usb_cameras(), start=1):
            if camera["id"] in existing:
                continue
            layout.setdefault("cameras", []).append(
                {
                    "id": camera["id"],
                    "label": f"ReefScope Camera #{index}",
                    "source_type": "usb",
                    "host_id": self.settings.host_id,
                    "tank_id": layout.get("active_tank_id", "tank-main"),
                    "angle": 315,
                    "radius": 1.18,
                    "position": _position_from_angle(315, 1.18),
                    "target": {"x": 0, "y": 0, "z": 0},
                    "fov": 44,
                    "enabled": True,
                }
            )
            changed = True
        if changed:
            self._write_layout(layout)

    def _ensure_3d_layout_fields(self, layout: dict[str, Any]) -> bool:
        changed = False
        for tank in layout.get("tanks", []):
            if "dimensions" not in tank:
                tank["dimensions"] = {"width": 1.0, "height": 1.0, "depth": 1.0}
                changed = True
            for item in tank.get("internals", []):
                if "z" not in item:
                    item["z"] = 0
                    changed = True
        for host in layout.get("hosts", []):
            if "position" not in host:
                host["position"] = {"x": float(host.get("x", 0)), "y": -0.75, "z": 1.2}
                changed = True
        for camera in layout.get("cameras", []):
            if "position" not in camera:
                camera["position"] = _position_from_angle(float(camera.get("angle", 0)), float(camera.get("radius", 1.08)))
                changed = True
            if "target" not in camera:
                camera["target"] = {"x": 0, "y": 0, "z": 0}
                changed = True
        return changed

    def _rotate(self, node_id: str) -> None:
        limit = self.settings.max_images_per_node
        if limit <= 0:
            return
        node_dir = self.settings.upload_dir / node_id
        images = sorted(node_dir.glob("*.jpg"), key=lambda path: path.stat().st_mtime, reverse=True)
        for old_image in images[limit:]:
            old_image.unlink(missing_ok=True)

    def _start_stream_session(self, state: dict[str, Any], node_id: str, command: dict[str, Any], delivered_at: str) -> None:
        duration = int(command.get("stream_seconds") or command.get("duration_seconds") or 30)
        duration = max(1, min(duration, 60))
        delivered_dt = _parse_iso(delivered_at)
        session_id = f"{node_id}_{delivered_dt.strftime('%Y%m%dT%H%M%S%fZ')}"
        session = {
            "node_id": node_id,
            "session_id": session_id,
            "requested_at": command.get("queued_at", delivered_at),
            "command_returned_at": delivered_at,
            "duration_seconds": duration,
            "expected_until": (delivered_dt + timedelta(seconds=duration)).isoformat(),
            "grace_until": (delivered_dt + timedelta(seconds=duration + 10)).isoformat(),
            "status": "waiting_for_frames",
            "frame_count": 0,
            "latest_frame": None,
            "frames": [],
        }
        state.setdefault("active_stream_sessions", {})[node_id] = session
        state.setdefault("stream_sessions", {})[session_id] = session

    def _record_stream_frame(self, node_id: str, filename: str, size_bytes: int) -> None:
        state = self._read_state()
        session = state.get("active_stream_sessions", {}).get(node_id)
        if not session:
            return
        self._refresh_stream_session_state(session)
        if session.get("status") == "ended":
            state.setdefault("active_stream_sessions", {}).pop(node_id, None)
            state.setdefault("stream_sessions", {})[session["session_id"]] = session
            self._write_state(state)
            return
        now = _now()
        frame = {
            "filename": filename,
            "url": f"/uploads/{node_id}/{filename}",
            "size_bytes": size_bytes,
            "received_at": now,
        }
        frames = list(session.get("frames") or [])
        frames.append(frame)
        session["latest_frame"] = frame
        session["frames"] = frames[-90:]
        session["last_frame_at"] = now
        session["frame_count"] = int(session.get("frame_count", 0)) + 1
        session["status"] = "active"
        state.setdefault("active_stream_sessions", {})[node_id] = session
        state.setdefault("stream_sessions", {})[session["session_id"]] = session
        self._write_state(state)

    def _session_frames_from_disk(self, node_id: str, session: dict[str, Any]) -> list[dict[str, Any]]:
        start_raw = str(session.get("command_returned_at") or session.get("requested_at") or "")
        end_raw = str(session.get("grace_until") or session.get("expected_until") or "")
        if not start_raw or not end_raw:
            return []
        try:
            start = _parse_iso(start_raw)
            end = _parse_iso(end_raw)
        except ValueError:
            return []
        frames = []
        for path in sorted((self.settings.upload_dir / node_id).glob(f"{node_id}_*.jpg")):
            try:
                captured_at = datetime.strptime(path.stem.removeprefix(f"{node_id}_"), "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if start <= captured_at <= end:
                frames.append(
                    {
                        "filename": path.name,
                        "url": f"/uploads/{node_id}/{path.name}",
                        "size_bytes": path.stat().st_size,
                        "received_at": captured_at.isoformat(),
                    }
                )
        return frames[-90:]

    def _refresh_stream_session_state(self, session: dict[str, Any]) -> bool:
        now = datetime.now(timezone.utc)
        grace_until = _parse_iso(str(session.get("grace_until") or session.get("expected_until") or _now()))
        last_frame = session.get("last_frame_at")
        if last_frame:
            last_frame_at = _parse_iso(str(last_frame))
            stale = now > last_frame_at + timedelta(seconds=8)
        else:
            stale = now > grace_until
        if now > grace_until or stale:
            if session.get("status") != "ended":
                session["status"] = "ended"
                session["ended_at"] = _now()
                return True
        return False

    def _tank_ids(self) -> list[str]:
        tank_ids = sorted({camera["tank_id"] for camera in self.consolidated_cameras() if camera.get("enabled", True) and camera.get("tank_id")})
        return tank_ids or ["tank-main"]

    def _validate_node_assignment(self, node_id: str, hub_id: str | None, require_hub_id: bool = False) -> None:
        if node_id not in self.settings.allowed_nodes:
            self._mark_misassigned(node_id, hub_id, "node_not_assigned_to_hub")
            raise ValueError(f"{node_id} is not assigned to {self.settings.hub_id}")
        if require_hub_id and not hub_id:
            self._mark_misassigned(node_id, hub_id, "missing_hub_id")
            return
        if hub_id and hub_id != self.settings.hub_id:
            self._mark_misassigned(node_id, hub_id, "hub_id_mismatch")
            raise ValueError(f"{node_id} advertised hub {hub_id}, expected {self.settings.hub_id}")

    def _mark_misassigned(self, node_id: str, hub_id: str | None, reason: str) -> None:
        state = self._read_state()
        state.setdefault("misassigned_nodes", {})[node_id] = {
            "node_id": node_id,
            "advertised_hub_id": hub_id,
            "expected_hub_id": self.settings.hub_id,
            "reason": reason,
            "last_seen": _now(),
        }
        self._write_state(state)

    def _normalize_command(self, command: dict[str, Any]) -> dict[str, Any]:
        if command.get("command") == "capture":
            return {"command": "capture"}
        duration = int(command.get("stream_seconds") or command.get("duration_seconds") or 30)
        duration = max(1, min(duration, 60))
        return {"command": "stream", "stream_seconds": duration, "duration_seconds": duration}


def create_ingest_app(config_path: str | Path | None = None) -> Flask:
    config = load_config(config_path or os.environ.get("SYNC_TANK_CONFIG"))
    settings = _settings_from_config(config.raw.get("ingest", {}), config.raw.get("tank_identity", {}))
    store = NodeStore(settings)
    hub_client = HubClient(config.tank_id, config.hub)
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
    app.config["SYNC_TANK_INGEST"] = {"settings": settings, "store": store, "hub_client": hub_client}

    @app.before_request
    def restrict_usb_feeds_to_wired_link() -> tuple[Response, int] | None:
        if not request.path.startswith("/api/usb/"):
            return None
        if _remote_allowed(request.remote_addr or "", settings.usb_feed_allowed_cidrs):
            return None
        return jsonify({"error": "USB camera feeds are only available on the wired display-link network"}), 403

    @app.route("/")
    def dashboard() -> str:
        return render_template("ingest.html", summary=store.summary())

    @app.route("/api/node/heartbeat", methods=["POST"])
    def heartbeat() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400
        try:
            node = store.heartbeat(payload)
            return jsonify({"ok": True, "node": node})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/node/<node_id>/command", methods=["GET"])
    def node_command(node_id: str) -> Response:
        try:
            command = store.pop_node_command(node_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if command is None:
            return Response(status=204)
        return jsonify(_command_response(command))

    @app.route("/api/node/<node_id>/command", methods=["POST"])
    def set_node_command(node_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400
        try:
            command = store.set_node_command(node_id, payload)
            return jsonify({"ok": True, "node_id": node_id, "hub_id": settings.hub_id, "command": command})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/node/<node_id>/stream", methods=["POST"])
    def request_node_stream(node_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400
        stream_seconds = int(payload.get("stream_seconds") or payload.get("duration_seconds") or 30)
        try:
            command = store.set_node_command(node_id, {"command": "stream", "stream_seconds": stream_seconds})
            return jsonify(
                {
                    "ok": True,
                    "node_id": node_id,
                    "hub_id": settings.hub_id,
                    "state": "waiting_for_node_wake",
                    "command": _command_response(command),
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/node/<node_id>/capture", methods=["POST"])
    @app.route("/api/cameras/<node_id>/capture", methods=["POST"])
    def request_node_capture(node_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        if payload and not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400
        try:
            command = store.set_node_command(node_id, {"command": "capture"})
            return jsonify(
                {
                    "ok": True,
                    "node_id": node_id,
                    "hub_id": settings.hub_id,
                    "state": "waiting_for_node_wake",
                    "note": "ESP32 will receive this command the next time it wakes and polls the edge node.",
                    "command": _command_response(command),
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/node/<node_id>/stream/status")
    def node_stream_status(node_id: str) -> Response:
        try:
            history = store.stream_history(node_id)
            pending = store.pending_commands().get(node_id)
            if pending and pending.get("command") == "stream":
                return jsonify(
                    {
                        "node_id": node_id,
                        "status": "pending",
                        "active": False,
                        "pending_command": _command_response(pending),
                        "pending_queued_at": pending.get("queued_at"),
                        "history": history,
                    }
                )
            session = store.stream_session(node_id)
            history = store.stream_history(node_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not session:
            return jsonify({"node_id": node_id, "status": "idle", "active": False, "history": history})
        return jsonify({"node_id": node_id, "active": session.get("status") != "ended", "session": session, "history": history})

    @app.route("/api/node/<node_id>/stream/latest")
    def node_stream_latest(node_id: str):
        try:
            path, session = store.latest_stream_frame(node_id)
            return send_file(path, mimetype="image/jpeg", max_age=0, etag=False, last_modified=None)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/api/pc-hub/payload", methods=["POST"])
    @app.route("/api/images/upload", methods=["POST"])
    def upload() -> Response:
        node_id = request.headers.get("X-Node-Id", "")
        content_type = request.headers.get("content-type", "")
        if "image/jpeg" not in content_type.lower():
            return jsonify({"error": "Content-Type must be image/jpeg"}), 415
        image_bytes = request.get_data()
        try:
            node = store.image_upload(node_id, image_bytes, request.headers)
            app.logger.info(
                "accepted JPEG upload path=%s node_id=%s client_ip=%s size_bytes=%s",
                request.path,
                node["node_id"],
                request.remote_addr,
                len(image_bytes),
            )
            return jsonify(
                {
                    "ok": True,
                    "node_id": node["node_id"],
                    "hub_id": node.get("hub_id", settings.hub_id),
                    "latest_image": node.get("latest_image"),
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/nodes")
    def nodes() -> Response:
        return jsonify(store.summary())

    @app.route("/api/systems/self-test", methods=["POST"])
    def systems_self_test() -> Response:
        payload = request.get_json(silent=True) or {}
        repair = bool(payload.get("repair", False))
        return jsonify(usb_camera_self_test(repair=repair, timeout=5))

    @app.route("/api/layout", methods=["GET"])
    def get_layout() -> Response:
        return jsonify(store.layout())

    @app.route("/api/layout", methods=["POST"])
    def save_layout() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400
        try:
            return jsonify({"ok": True, "layout": store.save_layout(payload)})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/node-config", methods=["GET"])
    def get_node_config() -> Response:
        return jsonify(store.node_config())

    @app.route("/api/node-config", methods=["POST"])
    def save_node_config() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400
        try:
            return jsonify({"ok": True, "node_config": store.save_node_config(payload)})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/hub-payload", methods=["GET"])
    def hub_payload() -> Response:
        return jsonify(store.hub_payload())

    @app.route("/api/pc-hub/payload", methods=["GET"])
    def pc_hub_payload() -> Response:
        return jsonify(store.pc_contract_payload(_lan_url(settings)))

    @app.route("/api/pc-hub/register-node", methods=["POST"])
    def pc_register_node() -> Response:
        client = _request_hub_client(hub_client, config.tank_id)
        return jsonify(client.register_node(store.pc_node_payload(_lan_url(settings))))

    @app.route("/api/pc-hub/heartbeat", methods=["POST"])
    def pc_heartbeat() -> Response:
        client = _request_hub_client(hub_client, config.tank_id)
        return jsonify(client.send_heartbeat(settings.host_id))

    @app.route("/api/pc-hub/register-cameras", methods=["POST"])
    def pc_register_cameras() -> Response:
        client = _request_hub_client(hub_client, config.tank_id)
        return jsonify(client.register_cameras(settings.host_id, store.pc_camera_inventory(_lan_url(settings))))

    @app.route("/api/pc-hub/sync", methods=["POST"])
    def pc_sync() -> Response:
        client = _request_hub_client(hub_client, config.tank_id)
        return jsonify(_sync_pc_hub(client, store, settings))

    @app.route("/api/usb/<camera_id>/snapshot")
    def usb_snapshot(camera_id: str):
        camera = next((item for item in store.usb_cameras() if item["id"] == camera_id), None)
        if not camera:
            return jsonify({"error": "USB camera not found"}), 404
        try:
            # A snapshot failure must not terminate the FFmpeg process serving a live view.
            frame = capture_usb_snapshot(camera["device"], timeout=5)
            return send_file(BytesIO(frame), mimetype="image/jpeg")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 503

    @app.route("/api/usb/<camera_id>/stream")
    def usb_stream(camera_id: str):
        camera = next((item for item in store.usb_cameras() if item["id"] == camera_id), None)
        if not camera:
            return jsonify({"error": "USB camera not found"}), 404
        device = str(camera.get("device", ""))
        width = int(config.cameras.get("usb", {}).get("preferred_width", 1280))
        height = int(config.cameras.get("usb", {}).get("preferred_height", 720))
        fps = int(config.cameras.get("usb", {}).get("preferred_fps", 10))

        def generate():
            import subprocess
            from time import sleep

            while True:
                yielded_any = False
                for cmd in usb_mjpeg_command_candidates(device, width, height, fps):
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    try:
                        while process.stdout:
                            chunk = process.stdout.read(4096)
                            if not chunk:
                                break
                            yielded_any = True
                            yield chunk
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=1)
                    if yielded_any:
                        break
                sleep(0.5 if yielded_any else 1.5)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/uploads/<node_id>/<filename>")
    def uploaded_image(node_id: str, filename: str):
        node_id = _clean_node_id(node_id)
        if not filename.endswith(".jpg") or "/" in filename:
            return jsonify({"error": "Invalid filename"}), 400
        if filename == "latest.jpg":
            node = store._node(node_id)
            latest = node.get("latest_image") or {}
            filename = latest.get("filename", "")
            if not filename:
                return jsonify({"error": "No latest image for node"}), 404
        return send_from_directory(settings.upload_dir / node_id, filename)

    return app


def _settings_from_config(raw: dict[str, Any], rig_profile: dict[str, Any] | None = None) -> IngestSettings:
    upload_dir = Path(raw.get("upload_dir", "test_uploads"))
    if not upload_dir.is_absolute():
        upload_dir = PROJECT_ROOT / upload_dir
    state_path = Path(raw.get("state_path", "config/ingest_state.json"))
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path
    layout_path = Path(raw.get("layout_path", "config/tank_layout.json"))
    if not layout_path.is_absolute():
        layout_path = PROJECT_ROOT / layout_path
    node_config_path = Path(raw.get("node_config_path", "config/node_config.json"))
    if not node_config_path.is_absolute():
        node_config_path = PROJECT_ROOT / node_config_path
    return IngestSettings(
        hub_id=str(raw.get("hub_id") or raw.get("host_id", "pi-sync-tank-01")),
        host_id=str(raw.get("host_id", "pi-sync-tank-01")),
        host_label=str(raw.get("host_label", "Sync Tank Pi")),
        bind=str(raw.get("bind", "0.0.0.0")),
        port=int(raw.get("port", 8080)),
        upload_dir=upload_dir,
        state_path=state_path,
        layout_path=layout_path,
        node_config_path=node_config_path,
        max_images_per_node=int(raw.get("max_images_per_node", 500)),
        expected_nodes=list(raw.get("expected_nodes", [])),
        allowed_nodes=list(raw.get("allowed_nodes") or raw.get("expected_nodes", [])),
        node_angles={str(node_id): float(angle) for node_id, angle in (raw.get("node_angles") or {}).items()},
        pc_sync_interval_seconds=int(raw.get("pc_sync_interval_seconds", 15)),
        public_url=str(raw.get("public_url", "")),
        camera_service_url=str(raw.get("camera_service_url", "")),
        usb_feed_allowed_cidrs=list(raw.get("usb_feed_allowed_cidrs") or ["127.0.0.0/8"]),
        rig_profile=dict(rig_profile or {}),
    )


def _clean_node_id(node_id: str) -> str:
    if not NODE_ID_PATTERN.match(node_id):
        raise ValueError("Invalid or missing node ID")
    return node_id


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _command_response(command: dict[str, Any]) -> dict[str, Any]:
    if command.get("command") == "capture":
        return {"command": "capture"}
    stream_seconds = int(command.get("stream_seconds") or command.get("duration_seconds") or 30)
    return {"command": "stream", "duration_seconds": stream_seconds, "stream_seconds": stream_seconds}


def _lan_url(settings: IngestSettings) -> str:
    if settings.public_url and not _is_placeholder_url(settings.public_url):
        return settings.public_url
    return request.host_url.rstrip("/")


def _is_placeholder_url(value: str) -> bool:
    try:
        hostname = (urlsplit(str(value)).hostname or "").upper()
    except ValueError:
        return True
    return not hostname or any(token in hostname for token in ("PRIVATE", "WIRED", "TANK_", "PC_", "SYNC_"))


def _reachable_service_url(configured_url: str, request_url: str, port: int) -> str:
    if configured_url and not _is_placeholder_url(configured_url):
        return configured_url.rstrip("/")
    parsed = urlsplit(request_url)
    hostname = parsed.hostname or "localhost"
    host = f"[{hostname}]" if ":" in hostname else hostname
    return urlunsplit((parsed.scheme or "http", f"{host}:{int(port)}", "", "", "")).rstrip("/")


def _remote_allowed(remote_addr: str, allowed_cidrs: list[str]) -> bool:
    try:
        remote = ipaddress.ip_address(remote_addr)
    except ValueError:
        return False
    if remote.is_loopback or remote.is_private or remote.is_link_local:
        return True
    for cidr in allowed_cidrs:
        try:
            if remote in ipaddress.ip_network(str(cidr), strict=False):
                return True
        except ValueError:
            continue
    return False


def _sync_pc_hub(hub_client: HubClient, store: NodeStore, settings: IngestSettings) -> dict[str, Any]:
    lan_url = _lan_url(settings)
    node = hub_client.register_node(store.pc_node_payload(lan_url))
    heartbeat = hub_client.send_heartbeat(settings.host_id)
    cameras = hub_client.register_cameras(settings.host_id, store.pc_camera_inventory(lan_url))
    return {"node": node, "heartbeat": heartbeat, "cameras": cameras}


def _request_hub_client(default_client: HubClient, tank_id: str) -> HubClient:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict) or not payload.get("base_url"):
        return default_client
    config = {
        **default_client.config,
        "enabled": True,
        "base_url": str(payload["base_url"]),
    }
    if payload.get("api_key") is not None:
        config["api_key"] = str(payload.get("api_key") or "")
    return HubClient(tank_id, config)


def _position_from_angle(angle_deg: float, radius: float) -> dict[str, float]:
    import math

    angle = math.radians(angle_deg - 90)
    return {"x": round(math.cos(angle) * radius, 3), "y": 0, "z": round(math.sin(angle) * radius, 3)}


if __name__ == "__main__":
    app = create_ingest_app()
    settings = app.config["SYNC_TANK_INGEST"]["settings"]
    app.run(host=settings.bind, port=settings.port, threaded=True)
