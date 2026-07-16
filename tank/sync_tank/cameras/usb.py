from __future__ import annotations

import glob
import re
import os
import signal
import subprocess
from pathlib import Path
from typing import Any


def list_video_devices() -> list[dict[str, Any]]:
    devices = sorted(glob.glob("/dev/video*"))
    names = _device_names()
    cameras = []
    for device in devices:
        name = names.get(device, f"USB Camera {device}")
        if not _is_user_camera(name):
            continue
        identity = _device_identity(device)
        if not identity.get("is_capture"):
            continue
        safe_id = Path(device).name.replace("video", "usb_")
        cameras.append(
            {
                "id": safe_id,
                "name": name,
                "source_type": "usb",
                "device": device,
                "stable_match": identity,
                "status": "online",
                "stream_url": f"/api/cameras/{safe_id}/stream",
                "snapshot_url": f"/api/cameras/{safe_id}/snapshot",
            }
        )
    return cameras


def _is_user_camera(name: str) -> bool:
    lowered = name.lower()
    excluded = ("pispbe", "rpi-hevc", "bcm2835-codec", "rpivid", "unicam")
    if any(token in lowered for token in excluded):
        return False
    return "camera" in lowered or "cam" in lowered or "usb" in lowered


def _device_identity(device: str) -> dict[str, Any]:
    properties = _udev_properties(device)
    devlinks = properties.get("DEVLINKS", "").split()
    by_id = next((link for link in devlinks if "/by-id/" in link), "")
    id_path = properties.get("ID_PATH", "")
    by_path = next((link for link in devlinks if id_path and f"/by-path/{id_path}-" in link), "")
    if not by_path:
        by_path = next((link for link in devlinks if "/by-path/" in link and "usb" in link), "")
    capabilities = properties.get("ID_V4L_CAPABILITIES", "")
    return {
        "by_id": by_id,
        "by_path": by_path,
        "id_path": id_path,
        "vendor_id": properties.get("ID_VENDOR_ID", ""),
        "model_id": properties.get("ID_MODEL_ID", ""),
        "serial": properties.get("ID_SERIAL", ""),
        "product": properties.get("ID_V4L_PRODUCT", ""),
        "is_capture": ":capture:" in capabilities,
    }


def _udev_properties(device: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={device}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return {"ID_V4L_CAPABILITIES": ":capture:"}
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return properties


def _device_names() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return {}

    names: dict[str, str] = {}
    current = None
    for line in result.stdout.splitlines():
        if line and not line.startswith("\t"):
            current = line.strip().rstrip(":")
        match = re.search(r"(/dev/video\d+)", line)
        if current and match:
            names[match.group(1)] = current
    return names


def capture_usb_snapshot(device: str, timeout: int = 5) -> bytes:
    preferred_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-framerate",
        "10",
        "-video_size",
        "640x480",
        "-i",
        device,
        "-frames:v",
        "1",
        "-f",
        "image2",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    result = subprocess.run(preferred_cmd, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0 or not result.stdout:
        fallback_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-i",
            device,
            "-frames:v",
            "1",
            "-f",
            "image2",
            "-vcodec",
            "mjpeg",
            "-",
        ]
        result = subprocess.run(fallback_cmd, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore") or "USB snapshot failed")
    return result.stdout


def usb_mjpeg_command_candidates(device: str, width: int, height: int, fps: int) -> list[list[str]]:
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2"]
    output = ["-f", "mpjpeg", "-boundary_tag", "frame", "-q:v", "5", "-"]
    sizes = [(width, height), (640, 480)]
    commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for candidate_width, candidate_height in sizes:
        for use_mjpeg in (True, False):
            cmd = [*base]
            if use_mjpeg:
                cmd.extend(["-input_format", "mjpeg"])
            cmd.extend(["-framerate", str(fps), "-video_size", f"{candidate_width}x{candidate_height}", "-i", device, *output])
            key = tuple(cmd)
            if key not in seen:
                commands.append(cmd)
                seen.add(key)

    fallback = [*base, "-i", device, *output]
    if tuple(fallback) not in seen:
        commands.append(fallback)
    return commands


def capture_usb_snapshot_with_repair(device: str, timeout: int = 5) -> bytes:
    try:
        return capture_usb_snapshot(device, timeout=timeout)
    except RuntimeError as exc:
        if "Device or resource busy" not in str(exc):
            raise
        clear_camera_holders(device)
        return capture_usb_snapshot(device, timeout=timeout)


def usb_camera_self_test(repair: bool = False, timeout: int = 5) -> dict[str, Any]:
    results = []
    for camera in list_video_devices():
        result = _test_camera(camera, timeout=timeout)
        if not result["ok"] and repair and result["busy"]:
            result["repair"] = clear_camera_holders(camera["device"])
            retry = _test_camera(camera, timeout=timeout)
            result["retry"] = retry
            result["ok"] = retry["ok"]
            result["jpeg_bytes"] = retry.get("jpeg_bytes", 0)
            result["error"] = retry.get("error", result.get("error", ""))
        results.append(result)
    return {
        "ok": all(item["ok"] for item in results),
        "repair_requested": repair,
        "cameras": results,
    }


def clear_camera_holders(device: str) -> dict[str, Any]:
    holders = camera_holders(device)
    cleared = []
    skipped = []
    for holder in holders:
        pid = int(holder["pid"])
        command = str(holder.get("command", ""))
        if _is_safe_camera_helper(command):
            _terminate_process(pid)
            cleared.append(holder)
        else:
            skipped.append(holder)
    return {"device": device, "cleared": cleared, "skipped": skipped}


def camera_holders(device: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(["fuser", device], check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return []
    holders = []
    for token in result.stdout.split():
        if not token.isdigit():
            continue
        pid = int(token)
        holders.append({"pid": pid, "command": _process_command(pid)})
    return holders


def _test_camera(camera: dict[str, Any], timeout: int) -> dict[str, Any]:
    result = {
        "camera_id": camera["id"],
        "device": camera["device"],
        "name": camera.get("name", camera["id"]),
        "stable_match": camera.get("stable_match", {}),
        "ok": False,
        "busy": False,
        "jpeg_bytes": 0,
    }
    try:
        frame = capture_usb_snapshot(camera["device"], timeout=timeout)
        result["ok"] = frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")
        result["jpeg_bytes"] = len(frame)
        if not result["ok"]:
            result["error"] = "Capture returned bytes that were not a complete JPEG"
    except Exception as exc:
        error = str(exc)
        result["error"] = error
        result["busy"] = "Device or resource busy" in error
        if result["busy"]:
            result["holders"] = camera_holders(camera["device"])
    return result


def _is_safe_camera_helper(command: str) -> bool:
    lowered = command.lower()
    return "ffmpeg" in lowered and "-f v4l2" in lowered and "/dev/video" in lowered


def _process_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _terminate_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    import time

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        return
