#!/usr/bin/env python3
"""Deterministic fake tank node for offline development and soak tests."""

import argparse
import base64
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


# Valid fallback JPEG for environments that intentionally omit OpenCV.
BASE_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAMABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDw+iiiuA7j/9k="
)
FIXTURES = ("empty", "decorations", "fish", "reflections", "bubbles", "plant-movement", "camera-movement",
            "single-target", "multiple-targets", "target-loss", "frozen", "malformed", "slow", "disconnected")


def fixture_jpeg(name, frame):
    if name == "malformed":
        return b"not-a-jpeg"
    try:
        import cv2
        import numpy as np
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:] = (44, 31, 16)
        # Stationary reef landmarks give the motion model a stable background.
        cv2.rectangle(image, (25, 160), (95, 225), (62, 92, 82), -1)
        cv2.circle(image, (245, 195), 38, (85, 72, 58), -1)
        if name in ("fish", "single-target", "multiple-targets"):
            x = 30 + (frame * 7) % 250
            cv2.ellipse(image, (x, 105), (28, 11), 0, 0, 360, (205, 190, 96), -1)
            cv2.fillPoly(image, [np.array([[x - 25, 105], [x - 43, 92], [x - 43, 118]], np.int32)], (205, 190, 96))
        if name == "multiple-targets":
            cv2.ellipse(image, (285 - (frame * 5) % 240, 65), (18, 8), 0, 0, 360, (150, 190, 210), -1)
        if name in ("bubbles", "plant-movement"):
            for index in range(12):
                cv2.circle(image, (20 + index * 25, 210 - ((frame * 4 + index * 17) % 190)), 3, (210, 210, 190), 1)
        if name == "reflections":
            cv2.line(image, ((frame * 9) % 320, 0), ((frame * 9 + 70) % 320, 240), (230, 230, 230), 8)
        if name == "camera-movement":
            image = np.roll(image, (frame * 4) % 80, axis=1)
        if name == "target-loss" and frame % 24 < 12:
            cv2.ellipse(image, (80 + frame * 5, 105), (24, 10), 0, 0, 360, (205, 190, 96), -1)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            return encoded.tobytes()
    except ImportError:
        pass
    marker = f"fixture={name};frame={frame}".encode()
    return BASE_JPEG[:-2] + b"\xff\xfe" + len(marker + b"xx").to_bytes(2, "big") + marker + BASE_JPEG[-2:]


def validate_raydar_pose(pan, tilt):
    pan, tilt = float(pan), float(tilt)
    if not (20 <= pan <= 160 and 45 <= tilt <= 125):
        raise ValueError("unsafe pose rejected")
    return {"pan": pan, "tilt": tilt}


class FakeState:
    def __init__(self, node_id, tank_id, host, port):
        self.node_id, self.tank_id, self.host, self.port = node_id, tank_id, host, port
        self.fixture = "fish"
        self.frame = 0
        self.latency = 0.0
        self.http_error = 0
        self.commands = []
        self.raydar = {"pan": 90.0, "tilt": 90.0}
        self.reeflex_running = False

    def cameras(self):
        base = f"http://{self.host}:{self.port}"
        return [{
            "camera_id": f"{self.node_id}-cam-{index}", "node_id": self.node_id, "tank_id": self.tank_id,
            "source_type": "usb_camera", "camera_type": "lighthouse_cam" if index == 0 else "endoscope_cam",
            "role_locked": True, "label": "Raydar" if index == 0 else f"Reel {index}", "status": "online",
            "snapshot_url": f"{base}/api/cameras/{self.node_id}-cam-{index}/snapshot",
            "stream_url": f"{base}/api/cameras/{self.node_id}-cam-{index}/stream",
        } for index in range(4)]

    def payload(self):
        base = f"http://{self.host}:{self.port}"
        return {
            "node_id": self.node_id, "node_type": "fake_tank_node", "tank_ids": [self.tank_id],
            "label": f"Fake {self.tank_id}", "status": "online", "replace": True, "cameras": self.cameras(),
            "control_urls": {
                "lighthouse_pose": f"{base}/api/controls/lighthouse/pose",
                "reeflex_idle_start": f"{base}/api/controls/reeflex/idle/start",
                "reeflex_idle_stop": f"{base}/api/controls/reeflex/idle/stop",
                "reeflex_stop": f"{base}/api/controls/reeflex/stop",
            },
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return

    def json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def jpeg(self):
        state = self.server.state
        if state.fixture == "disconnected":
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE); return
        if state.fixture == "slow":
            time.sleep(max(1.0, state.latency))
        state.frame += 0 if state.fixture == "frozen" else 1
        data = b"not-a-jpeg" if state.fixture == "malformed" else fixture_jpeg(state.fixture, state.frame)
        self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("ETag", f'"{state.fixture}-{state.frame}"'); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        state = self.server.state
        if state.latency: time.sleep(state.latency)
        if state.http_error: self.send_error(state.http_error); return
        if path in ("/api/pc-hub/payload", "/api/payload"):
            self.json(state.payload()); return
        if path.endswith("/snapshot"):
            self.jpeg(); return
        if path.endswith("/stream"):
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            try:
                for _ in range(60):
                    state.frame += 1; data = fixture_jpeg(state.fixture, state.frame)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n" + f"Content-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n"); self.wfile.flush(); time.sleep(.2)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        if path == "/api/fake/state":
            self.json({"fixture": state.fixture, "frame": state.frame, "commands": state.commands, "raydar": state.raydar, "reeflex_running": state.reeflex_running}); return
        self.send_error(404)

    def do_POST(self):
        state = self.server.state; path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0)); data = json.loads(self.rfile.read(length) or b"{}")
        if path == "/api/fake/config":
            if data.get("fixture") not in (None, *FIXTURES): return self.json({"error": "invalid fixture"}, 400)
            for key in ("fixture", "latency", "http_error"):
                if key in data: setattr(state, key, data[key])
            return self.json({"status": "ok"})
        if path.endswith("lighthouse/pose"):
            try: state.raydar = validate_raydar_pose(data.get("pan", -1), data.get("tilt", -1))
            except ValueError as exc: return self.json({"error": str(exc)}, 422)
            state.commands.append({"at": time.time(), "path": path, **data}); return self.json({"status": "ok", **state.raydar})
        if "reeflex" in path:
            state.reeflex_running = path.endswith("/start"); state.commands.append({"at": time.time(), "path": path, **data}); return self.json({"status": "ok", "running": state.reeflex_running})
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=18081); parser.add_argument("--node-id", default="fake-tank-1"); parser.add_argument("--tank-id", default="tank-1")
    args = parser.parse_args(); server = ThreadingHTTPServer((args.host, args.port), Handler); server.state = FakeState(args.node_id, args.tank_id, args.host, args.port)
    print(f"Fake tank {args.tank_id} at http://{args.host}:{args.port}"); server.serve_forever()


if __name__ == "__main__": main()
