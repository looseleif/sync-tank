from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sync_tank.yaml"
DEFAULT_IDENTITY_PATH = PROJECT_ROOT / "config" / "tank_identity.yaml"


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
    if identity_path.exists():
        with identity_path.open("r", encoding="utf-8") as handle:
            identity = yaml.safe_load(handle) or {}
        raw = _apply_identity(raw, identity)
    return AppConfig(raw=raw, path=config_path)


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
