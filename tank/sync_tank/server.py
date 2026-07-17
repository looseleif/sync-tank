from __future__ import annotations

import ipaddress
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, redirect, render_template, request, send_file
try:
    from flask_cors import CORS
except ImportError:  # Keep the Pi app runnable before optional deps are installed.
    CORS = None

from sync_tank.arm import ArmController
from sync_tank.cameras.esp32 import discover_esp32_cameras, fetch_http_snapshot
from sync_tank.cameras.registry import CameraRegistry
from sync_tank.cameras.usb import capture_usb_snapshot_with_repair, list_video_devices, usb_camera_self_test, usb_mjpeg_command_candidates
from sync_tank.config import PROJECT_ROOT, AppConfig, load_config
from sync_tank.uplink import HubClient


def create_app(config_path: str | Path | None = None) -> Flask:
    config = load_config(config_path or os.environ.get("SYNC_TANK_CONFIG"))
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
    if CORS:
        CORS(app)

    registry_path = config.resolve_path(config.cameras.get("registry_path", "config/cameras.json"))
    registry = CameraRegistry(registry_path)
    arm = ArmController(config.arm)
    hub = HubClient(config.tank_id, config.hub)

    app.config["SYNC_TANK"] = {
        "config": config,
        "registry": registry,
        "arm": arm,
        "hub": hub,
    }

    @app.before_request
    def restrict_usb_feeds_to_wired_link() -> tuple[Response, int] | None:
        allowed_cidrs = list(config.raw.get("ingest", {}).get("usb_feed_allowed_cidrs") or ["127.0.0.0/8", "REDACTED_PRIVATE_IP/24"])
        control_paths = ("/api/arm", "/api/servo", "/api/lighthouse", "/api/reeflex")
        if request.path.startswith(control_paths):
            if _remote_allowed(request.remote_addr or "", allowed_cidrs):
                return None
            return jsonify({"error": "Servo controls are only available on the wired display-link network"}), 403
        if not request.path.startswith("/api/cameras/") or not (request.path.endswith("/snapshot") or request.path.endswith("/stream")):
            return None
        camera_id = request.path.split("/")[3]
        camera = registry.get(camera_id)
        if not camera or camera.get("source_type") != "usb":
            return None
        if _remote_allowed(request.remote_addr or "", allowed_cidrs):
            return None
        return jsonify({"error": "USB camera feeds are only available on the wired display-link network"}), 403

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/status")
    def status() -> Response:
        _refresh_usb(registry)
        return jsonify(
            {
                "tank_id": config.tank_id,
                "arm": arm.status(),
                "cameras": registry.list(),
                "hub": hub.status(),
            }
        )

    @app.route("/api/arm")
    def arm_status() -> Response:
        return jsonify(arm.status())

    @app.route("/api/arm/servo/<servo_id>", methods=["POST"])
    def set_servo(servo_id: str) -> Response:
        payload = request.get_json(silent=True) or {}
        if "angle" not in payload:
            return jsonify({"error": "Missing angle"}), 400
        try:
            return jsonify(arm.set_angle(servo_id, float(payload["angle"]), reject_out_of_range=True))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc) or "Angle must be numeric"}), 400

    @app.route("/api/servo/channel", methods=["POST"])
    def set_servo_channel() -> Response:
        payload = request.get_json(silent=True) or {}
        if "channel" not in payload or "angle" not in payload:
            return jsonify({"error": "Missing channel or angle"}), 400
        try:
            result = arm.set_channel(
                int(payload["channel"]),
                float(payload["angle"]),
                device_id=payload.get("device_id"),
                joint=payload.get("joint"),
            )
            return jsonify({"status": "ok", "result": result, "arm": arm.status()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc) or "Channel and angle must be numeric"}), 400

    @app.route("/api/lighthouse/pose", methods=["POST"])
    def set_lighthouse_pose() -> Response:
        payload = request.get_json(silent=True) or {}
        device_id = str(payload.get("device_id") or "lighthouse-001")
        pose = {key: payload[key] for key in ("pan", "tilt") if key in payload}
        if not pose:
            return jsonify({"error": "Missing pan or tilt"}), 400
        try:
            return jsonify(arm.set_device_pose(device_id, pose))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/reeflex/pose", methods=["POST"])
    def set_reeflex_pose() -> Response:
        payload = request.get_json(silent=True) or {}
        device_id = str(payload.get("device_id") or "reeflex-001")
        pose = {key: payload[key] for key in ("base", "shoulder", "elbow") if key in payload}
        if not pose:
            return jsonify({"error": "Missing base, shoulder, or elbow"}), 400
        try:
            return jsonify(arm.set_device_pose(device_id, pose))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/reeflex/idle", methods=["GET"])
    def reeflex_idle_status() -> Response:
        return jsonify({"arm": arm.status(), "idle": arm.status().get("idle", {})})

    @app.route("/api/reeflex/idle/start", methods=["POST"])
    def start_reeflex_idle() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(
                arm.start_idle_scan(
                    str(payload.get("device_id") or "reeflex-001"),
                    center=payload.get("center") if isinstance(payload.get("center"), dict) else None,
                    amplitude=float(payload.get("amplitude", 8.0)),
                    period_seconds=float(payload.get("period_seconds", 9.0)),
                    step_seconds=float(payload.get("step_seconds", 0.35)),
                )
            )
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/reeflex/idle/stop", methods=["POST"])
    def stop_reeflex_idle() -> Response:
        return jsonify(arm.stop_idle(reason="operator_stop"))

    @app.route("/api/arm/stop", methods=["POST"])
    def stop_arm() -> Response:
        return jsonify(arm.stop())

    @app.route("/api/cameras")
    def cameras() -> Response:
        _refresh_usb(registry)
        return jsonify(registry.list())

    @app.route("/api/cameras/self-test", methods=["POST"])
    def camera_self_test() -> Response:
        payload = request.get_json(silent=True) or {}
        repair = bool(payload.get("repair", False))
        result = usb_camera_self_test(repair=repair, timeout=int(config.cameras.get("usb", {}).get("snapshot_timeout_seconds", 5)))
        for camera in result["cameras"]:
            registry.mark_status(camera["camera_id"], "online" if camera["ok"] else "offline")
        return jsonify(result)

    @app.route("/api/cameras/discover", methods=["POST"])
    def discover_cameras() -> Response:
        _refresh_usb(registry)
        discovered = discover_esp32_cameras(config.cameras.get("esp32", {}))
        saved = [registry.upsert(camera) for camera in discovered]
        return jsonify({"count": len(saved), "cameras": saved})

    @app.route("/api/cameras/<camera_id>/snapshot")
    def camera_snapshot(camera_id: str) -> Response:
        camera = registry.get(camera_id)
        if not camera:
            return jsonify({"error": "Camera not found"}), 404
        try:
            frame = _capture_snapshot(camera, config)
            registry.mark_status(camera_id, "online")
            return send_file(BytesIO(frame), mimetype="image/jpeg")
        except Exception as exc:
            registry.mark_status(camera_id, "offline")
            return jsonify({"error": str(exc)}), 503

    @app.route("/api/cameras/<camera_id>/stream")
    def camera_stream(camera_id: str) -> Response:
        camera = registry.get(camera_id)
        if not camera:
            return jsonify({"error": "Camera not found"}), 404
        if camera.get("source_type") == "esp32" and camera.get("stream_url", "").startswith("http"):
            return redirect(camera["stream_url"])
        if camera.get("source_type") == "usb":
            return _usb_mjpeg_response(camera, config)
        snapshot_url = camera.get("snapshot_url")
        if snapshot_url:
            return redirect(snapshot_url)
        return jsonify({"error": "No stream available"}), 404

    @app.route("/api/uplink/test", methods=["POST"])
    def uplink_test() -> Response:
        return jsonify(hub.test())

    @app.route("/api/uplink/cameras/<camera_id>/snapshot", methods=["POST"])
    def uplink_snapshot(camera_id: str) -> Response:
        camera = registry.get(camera_id)
        if not camera:
            return jsonify({"error": "Camera not found"}), 404
        payload = request.get_json(silent=True) or {}
        try:
            frame = _capture_snapshot(camera, config)
            return jsonify(hub.send_frame(camera, frame, note=str(payload.get("note", ""))))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 503

    @app.route("/api/uplink/cameras/<camera_id>/start", methods=["POST"])
    def uplink_start(camera_id: str) -> Response:
        camera = registry.get(camera_id)
        if not camera:
            return jsonify({"error": "Camera not found"}), 404
        return jsonify(hub.start_stream(camera))

    @app.route("/api/uplink/cameras/<camera_id>/stop", methods=["POST"])
    def uplink_stop(camera_id: str) -> Response:
        return jsonify(hub.stop_stream(camera_id))

    return app


def _refresh_usb(registry: CameraRegistry) -> None:
    current_usb = list_video_devices()
    current_ids = {camera["id"] for camera in current_usb}
    cameras = registry.load()
    pruned = {
        camera_id: camera
        for camera_id, camera in cameras.items()
        if camera.get("source_type") != "usb" or camera_id in current_ids
    }
    if pruned != cameras:
        registry.save(pruned)
    for camera in current_usb:
        registry.upsert(camera)


def _remote_allowed(remote_addr: str, allowed_cidrs: list[str]) -> bool:
    try:
        remote = ipaddress.ip_address(remote_addr)
    except ValueError:
        return False
    for cidr in allowed_cidrs:
        try:
            if remote in ipaddress.ip_network(str(cidr), strict=False):
                return True
        except ValueError:
            continue
    return False


def _capture_snapshot(camera: dict[str, Any], config: AppConfig) -> bytes:
    if camera.get("source_type") == "usb":
        timeout = int(config.cameras.get("usb", {}).get("snapshot_timeout_seconds", 5))
        return capture_usb_snapshot_with_repair(str(camera["device"]), timeout=timeout)
    snapshot_url = camera.get("snapshot_url") or camera.get("discovered_url")
    if snapshot_url and str(snapshot_url).startswith("http"):
        return fetch_http_snapshot(str(snapshot_url), timeout=int(config.hub.get("timeout_seconds", 5)))
    raise RuntimeError("Camera has no snapshot source")


def _usb_mjpeg_response(camera: dict[str, Any], config: AppConfig) -> Response:
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


if __name__ == "__main__":
    app = create_app()
    cfg = app.config["SYNC_TANK"]["config"]
    app.run(host=cfg.host.get("bind", "0.0.0.0"), port=int(cfg.host.get("port", 5050)), threaded=True)
