from __future__ import annotations

import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests


def discover_esp32_cameras(config: dict[str, Any]) -> list[dict[str, Any]]:
    ranges = config.get("discovery_ranges") or []
    endpoints = config.get("endpoints") or ["/stream", "/capture", "/"]
    timeout = float(config.get("timeout_seconds", 1.5))
    hosts = []
    for network in ranges:
        try:
            hosts.extend(str(ip) for ip in ipaddress.ip_network(network, strict=False).hosts())
        except ValueError:
            continue

    found: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_probe_host, host, endpoints, timeout): host for host in hosts}
        for future in as_completed(futures):
            camera = future.result()
            if camera:
                found.append(camera)
    return found


def _probe_host(host: str, endpoints: list[str], timeout: float) -> dict[str, Any] | None:
    if not _has_http_port(host, timeout):
        return None
    for endpoint in endpoints:
        url = f"http://{host}{endpoint}"
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            content_type = response.headers.get("content-type", "").lower()
            if response.status_code < 400 and _looks_like_camera_response(endpoint, content_type, response):
                response.close()
                camera_id = f"esp32_{host.replace('.', '_')}"
                stream_url = url if "multipart" in content_type or "stream" in endpoint else None
                snapshot_url = url if not stream_url else f"http://{host}/capture"
                return {
                    "id": camera_id,
                    "name": f"ESP32 Floater {host}",
                    "source_type": "esp32",
                    "host": host,
                    "status": "online",
                    "stream_url": stream_url,
                    "snapshot_url": snapshot_url,
                    "discovered_url": url,
                }
            response.close()
        except requests.RequestException:
            continue
    return None


def _has_http_port(host: str, timeout: float) -> bool:
    try:
        with socket.create_connection((host, 80), timeout=timeout):
            return True
    except OSError:
        return False


def _looks_like_camera_response(endpoint: str, content_type: str, response: requests.Response) -> bool:
    if "image/" in content_type or "multipart" in content_type:
        return True
    lowered = endpoint.lower()
    if any(token in lowered for token in ("stream", "capture", "jpg", "video", "cam-")):
        return response.status_code < 400
    return False


def fetch_http_snapshot(url: str, timeout: int = 5) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content
