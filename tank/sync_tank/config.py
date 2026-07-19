from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sync_tank.yaml"
DEFAULT_IDENTITY_PATH = PROJECT_ROOT / "config" / "tank_identity.yaml"
DEFAULT_PROFILES_PATH = PROJECT_ROOT / "config" / "tank_profiles.yaml"
NODE_PROFILE_BY_ID = {"tank-pi-001": "tank1-raydar", "tank-pi-002": "tank2-reeflex"}


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def tank_id(self) -> str:
        return str(self.raw.get("tank_id", "sync-tank-01"))

    @property
    def host(self) -> dict[str, Any]:
        return self.raw.get("host", {})

    @property
    def arm(self) -> dict[str, Any]:
        return self.raw.get("arm", {})

    @property
    def cameras(self) -> dict[str, Any]:
        return self.raw.get("cameras", {})

    @property
    def hub(self) -> dict[str, Any]:
        return self.raw.get("hub", {})

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    identity_path = config_path.parent / "tank_identity.yaml"
    identity: dict[str, Any] = {}
    if identity_path.exists():
        with identity_path.open("r", encoding="utf-8") as handle:
            identity = yaml.safe_load(handle) or {}
    profiles_path = config_path.parent / "tank_profiles.yaml"
    profiles = _load_profiles(profiles_path)
    profile_id = _select_profile_id(config_path, raw, identity, profiles)
    if profile_id in profiles:
        identity = _identity_from_profile(profile_id, profiles[profile_id])
        raw["selected_profile"] = profile_id
        raw["autonomy"] = dict(profiles[profile_id].get("autonomy") or {})
    if identity:
        raw = _apply_identity(raw, identity)
        raw["arm"] = arm_config_for_identity(identity, raw.get("arm") or {})
    return AppConfig(raw=raw, path=config_path)


def _load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return dict((yaml.safe_load(handle) or {}).get("profiles") or {})


def _select_profile_id(config_path: Path, raw: dict[str, Any], identity: dict[str, Any], profiles: dict[str, Any]) -> str:
    requested = str(os.environ.get("SYNC_TANK_PROFILE", "")).strip()
    role_path = config_path.parent / "node_role"
    if not requested and role_path.exists():
        requested = role_path.read_text(encoding="utf-8").strip()
    if requested in profiles:
        return requested

    node_config_path = Path(str((raw.get("ingest") or {}).get("node_config_path") or "node_config.json"))
    candidates = [node_config_path] if node_config_path.is_absolute() else [config_path.parent / "node_config.json", PROJECT_ROOT / node_config_path]
    for candidate in candidates:
        try:
            runtime = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        node_id = str((runtime.get("node") or {}).get("id") or "")
        if node_id in NODE_PROFILE_BY_ID:
            return NODE_PROFILE_BY_ID[node_id]
        runtime_profile = str((runtime.get("profile") or {}).get("id") or "")
        if runtime_profile in profiles:
            return runtime_profile

    node_id = str((identity.get("node") or {}).get("id") or "")
    if node_id in NODE_PROFILE_BY_ID:
        return NODE_PROFILE_BY_ID[node_id]
    identity_profile = str((identity.get("profile") or {}).get("id") or "")
    return identity_profile if identity_profile in profiles else ""


def _identity_from_profile(profile_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    inventory = profile.get("inventory") or {}
    return {
        "tank": dict(profile.get("tank") or {}),
        "node": {**dict(profile.get("node") or {}), "role": "raspberry_pi_tank_node"},
        "esp32": {
            **dict(profile.get("esp32") or {}),
            "allowed_nodes": list((profile.get("esp32") or {}).get("expected_nodes") or []),
        },
        "inventory": dict(inventory),
        "profile": {"id": profile_id, "role_split": dict(profile.get("role_split") or {})},
    }


def arm_config_for_identity(identity: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    role = (identity.get("profile") or {}).get("role_split") or {}
    arm = {
        "backend": existing.get("backend", "pca9685"),
        "pca9685": dict(existing.get("pca9685") or {"address": "0x40", "bus": 1, "frequency_hz": 50, "min_tick": 150, "max_tick": 600}),
        "disable_pwm_after_move": bool(existing.get("disable_pwm_after_move", False)),
        "movement_delay_seconds": float(existing.get("movement_delay_seconds", 0.35)),
    }
    if role.get("lighthouse") and not role.get("reeflex"):
        arm["servos"] = {
            "lighthouse_pan": {"name": "Raydar Pan", "channel": 1, "min_angle": 20, "max_angle": 160, "neutral_angle": 90, "min_pulse_width": 0.0005, "max_pulse_width": 0.0025},
            "lighthouse_tilt": {"name": "Raydar Tilt", "channel": 0, "min_angle": 45, "max_angle": 125, "neutral_angle": 90, "min_pulse_width": 0.0005, "max_pulse_width": 0.0025},
        }
        arm["devices"] = {"lighthouse-001": {"type": "lighthouse", "joints": {"pan": "lighthouse_pan", "tilt": "lighthouse_tilt"}}}
    elif role.get("reeflex") and not role.get("lighthouse"):
        arm["servos"] = {
            "reeflex_base": {"name": "Reeflex Base", "channel": 0, "min_angle": 20, "max_angle": 160, "neutral_angle": 90, "min_pulse_width": 0.0005, "max_pulse_width": 0.0025},
            "reeflex_shoulder": {"name": "Reeflex Shoulder", "channel": 1, "min_angle": 45, "max_angle": 135, "neutral_angle": 90, "min_pulse_width": 0.0005, "max_pulse_width": 0.0025},
            "reeflex_elbow": {"name": "Reeflex Elbow", "channel": 2, "min_angle": 35, "max_angle": 145, "neutral_angle": 90, "min_pulse_width": 0.0005, "max_pulse_width": 0.0025},
        }
        arm["devices"] = {"reeflex-001": {"type": "reeflex", "joints": {"base": "reeflex_base", "shoulder": "reeflex_shoulder", "elbow": "reeflex_elbow"}}}
    else:
        arm["servos"] = {}
        arm["devices"] = {}
    return arm


def _apply_identity(raw: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    merged = dict(raw)
    merged["tank_identity"] = identity

    tank = identity.get("tank") or {}
    node = identity.get("node") or {}
    network = identity.get("network") or {}
    esp32 = identity.get("esp32") or {}

    if tank.get("id"):
        merged["tank_id"] = str(tank["id"])

    ingest = dict(merged.get("ingest") or {})
    node_id = str(node.get("id") or ingest.get("host_id") or ingest.get("hub_id") or "")
    if node_id:
        ingest["host_id"] = node_id
        ingest["hub_id"] = node_id
    if node.get("label"):
        ingest["host_label"] = str(node["label"])
    if network.get("public_url"):
        ingest["public_url"] = str(network["public_url"])
    if network.get("camera_service_url"):
        ingest["camera_service_url"] = str(network["camera_service_url"])
    if network.get("usb_feed_allowed_cidrs"):
        ingest["usb_feed_allowed_cidrs"] = list(network["usb_feed_allowed_cidrs"])
    if esp32.get("expected_nodes") is not None:
        expected_nodes = [str(item) for item in esp32.get("expected_nodes") or []]
        ingest["expected_nodes"] = expected_nodes
        ingest["allowed_nodes"] = [str(item) for item in esp32.get("allowed_nodes") or expected_nodes]
    if esp32.get("node_angles"):
        ingest["node_angles"] = {str(key): float(value) for key, value in dict(esp32["node_angles"]).items()}
    merged["ingest"] = ingest
    return merged
