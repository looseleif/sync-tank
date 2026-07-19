"""Offline-first wildlife observation and safe rig-control primitives.

The controller deliberately has no hardware discovery of its own.  It only uses
URLs advertised by tank nodes and it becomes Unavailable when those URLs vanish.
That keeps the same code testable with the fake tank server and safe on a Sync Pi.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror, request as urlrequest


AI_DISCLOSURE = "Sends this captured image to OpenAI for analysis"
AI_LABEL = "AI field note — identification not confirmed"
SIGHTING_LABELS = ("Fish", "Shrimp", "Snail", "Coral", "Unknown")
STRUCTURE_TYPES = ("block", "slab", "rounded-rock", "pillar", "arch", "mound")
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def frame_revision(headers: Dict[str, str], jpeg: bytes) -> Tuple[str, str]:
    """Return a stable revision, preferring cheap HTTP validators."""
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    if lowered.get("etag"):
        return "etag", lowered["etag"]
    if lowered.get("last-modified"):
        return "last-modified", lowered["last-modified"]
    return "sha256", hashlib.sha256(jpeg).hexdigest()


def is_jpeg(data: bytes) -> bool:
    return 4 <= len(data) <= MAX_IMAGE_BYTES and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def normalize_structure(item: Dict) -> Dict:
    """Validate and grid-snap a persisted simulator landmark."""
    if item.get("structure_type") not in STRUCTURE_TYPES:
        raise ValueError("invalid structure type")
    placement = dict(item.get("placement") or {})
    position = dict(placement.get("position") or {})
    if not all(key in position for key in ("x", "y", "z")):
        raise ValueError("structure position is required")
    snapped = {key: round(max(0.0, min(1.0, float(position[key]))) / 0.05) * 0.05 for key in ("x", "y", "z")}
    placement.update({"placed": True, "position": snapped})
    return {
        **item,
        "item_type": "structure_shape",
        "placement": placement,
        "rotation": float(item.get("rotation", 0)) % 360,
        "scale": max(0.25, min(3.0, float(item.get("scale", 1)))),
        "color": str(item.get("color") or "#698f88")[:32],
        "label": str(item.get("label") or item["structure_type"].replace("-", " "))[:80],
    }


def raydar_waypoints(center_pan: float = 90, center_tilt: float = 90,
                      pan_span: float = 25, tilt_span: float = 10,
                      count: int = 12) -> List[Dict[str, float]]:
    return [
        {
            "pan": round(center_pan + pan_span * math.cos(2 * math.pi * i / count), 3),
            "tilt": round(center_tilt + tilt_span * math.sin(2 * math.pi * i / count), 3),
        }
        for i in range(count)
    ]


def proportional_centering(centroid: Tuple[float, float], current: Tuple[float, float],
                           limits: Dict[str, float], dead_zone: float = 0.12,
                           max_step: float = 2.0, gain: float = 5.0) -> Tuple[float, float, bool]:
    """Compute one bounded pan/tilt update from normalized image coordinates."""
    x, y = centroid
    pan, tilt = current
    dx, dy = x - 0.5, y - 0.5
    centered = abs(dx) <= dead_zone / 2 and abs(dy) <= dead_zone / 2
    if centered:
        return pan, tilt, True
    pan_step = max(-max_step, min(max_step, dx * gain))
    tilt_step = max(-max_step, min(max_step, -dy * gain))
    next_pan = max(limits.get("min_pan", 0), min(limits.get("max_pan", 180), pan + pan_step))
    next_tilt = max(limits.get("min_tilt", 0), min(limits.get("max_tilt", 180), tilt + tilt_step))
    return round(next_pan, 3), round(next_tilt, 3), False


def select_best_capture(frames: Iterable[Dict]) -> Optional[Dict]:
    """Choose the sharpest centered frame from already-computed burst scores."""
    candidates = list(frames)
    if not candidates:
        return None
    return max(candidates, key=lambda frame: float(frame.get("sharpness", 0)) * 0.65 + float(frame.get("center_score", 0)) * 0.35)


class MotionTracker:
    """Lightweight persistence and centroid smoothing independent of OpenCV."""

    def __init__(self, persistence_frames: int = 3, smoothing: float = 0.35) -> None:
        self.persistence_frames = persistence_frames
        self.smoothing = smoothing
        self.frames = 0
        self.centroid: Optional[Tuple[float, float]] = None
        self.last_seen_at = 0.0

    def update(self, candidates: Iterable[Dict], now: Optional[float] = None) -> Optional[Dict]:
        now = time.time() if now is None else now
        valid = [c for c in candidates if 0 < float(c.get("area", 0)) < 0.35]
        if not valid:
            self.frames = 0
            return None
        target = max(valid, key=lambda c: float(c.get("area", 0)) * (0.5 + float(c.get("persistence", 1))))
        point = (float(target["x"]), float(target["y"]))
        if self.centroid is None:
            self.centroid = point
        else:
            a = self.smoothing
            self.centroid = (self.centroid[0] * (1 - a) + point[0] * a,
                             self.centroid[1] * (1 - a) + point[1] * a)
        self.frames += 1
        self.last_seen_at = now
        return {**target, "x": self.centroid[0], "y": self.centroid[1], "persistent": self.frames >= self.persistence_frames}

    def lost_for(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, now - self.last_seen_at) if self.last_seen_at else float("inf")


class OpenCVMotionAnalyzer:
    """320x240 background subtraction, loaded only when OpenCV is installed."""

    def __init__(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            self.cv2 = self.np = self.model = None
            return
        self.cv2, self.np = cv2, np
        self.model = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=22, detectShadows=True)

    @property
    def available(self) -> bool:
        return self.model is not None

    def analyze_jpeg(self, jpeg: bytes, learning: bool = True) -> List[Dict]:
        if not self.available or not is_jpeg(jpeg):
            return []
        frame = self.cv2.imdecode(self.np.frombuffer(jpeg, dtype=self.np.uint8), self.cv2.IMREAD_COLOR)
        if frame is None:
            return []
        frame = self.cv2.resize(frame, (320, 240))
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        gray = self.cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.model.apply(gray, learningRate=-1 if learning else 0)
        mask = self.cv2.threshold(mask, 220, 255, self.cv2.THRESH_BINARY)[1]
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, self.np.ones((3, 3), self.np.uint8))
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = self.cv2.contourArea(contour) / (320 * 240)
            if not 0.0008 <= area <= 0.20:
                continue
            x, y, width, height = self.cv2.boundingRect(contour)
            candidates.append({"x": (x + width / 2) / 320, "y": (y + height / 2) / 240, "area": area,
                               "box": {"x": x / 320, "y": y / 240, "width": width / 320, "height": height / 240}})
        return candidates


@dataclass
class RigState:
    state: str = "STOP"
    enabled: bool = False
    detail: str = "Controller stopped"
    tank_id: Optional[str] = None
    camera_id: Optional[str] = None
    updated_at: float = 0.0
    last_command_at: Optional[float] = None

    def payload(self) -> Dict:
        return dict(self.__dict__)


class VisionController:
    """Safety state machine; callers may drive ``tick_raydar`` at about 5 FPS."""

    def __init__(self, control_url: Callable[..., Optional[str]],
                 post_json: Optional[Callable[[str, Dict], Dict]] = None,
                 frame_source: Optional[Callable[[str], Optional[bytes]]] = None,
                 capture_callback: Optional[Callable[[Dict], None]] = None) -> None:
        self.control_url = control_url
        self.post_json = post_json or self._post_json
        self.lock = threading.RLock()
        self.frame_source = frame_source
        self.capture_callback = capture_callback
        self.analyzer = OpenCVMotionAnalyzer()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self.raydar = RigState()
        self.reeflex = RigState()
        self.tracker = MotionTracker()
        center_pan = float(os.environ.get("SYNC_RAYDAR_CENTER_PAN", "90"))
        center_tilt = float(os.environ.get("SYNC_RAYDAR_CENTER_TILT", "90"))
        self.waypoints = raydar_waypoints(
            center_pan, center_tilt,
            float(os.environ.get("SYNC_RAYDAR_PAN_SPAN", "25")),
            float(os.environ.get("SYNC_RAYDAR_TILT_SPAN", "10")),
        )
        self.waypoint_index = 0
        self.pose = (center_pan, center_tilt)
        self.tracking_started_at: Optional[float] = None
        self.centered_since: Optional[float] = None
        self.last_target: Optional[Dict] = None
        self._auto_captured_for_track = False
        self.limits = {
            "min_pan": float(os.environ.get("SYNC_RAYDAR_MIN_PAN", "20")),
            "max_pan": float(os.environ.get("SYNC_RAYDAR_MAX_PAN", "160")),
            "min_tilt": float(os.environ.get("SYNC_RAYDAR_MIN_TILT", "45")),
            "max_tilt": float(os.environ.get("SYNC_RAYDAR_MAX_TILT", "125")),
        }

    @staticmethod
    def _post_json(url: str, payload: Dict) -> Dict:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=3) as response:
            return json.loads(response.read(512_000) or b"{}")

    def status(self) -> Dict:
        with self.lock:
            return {
                "raydar": {**self.raydar.payload(), "target": self.last_target, "display_crop_max": 1.5},
                "reeflex": self.reeflex.payload(),
                "analysis": {"resolution": [320, 240], "fps": 5, "classification": "interesting motion only"},
                "safety": {"manual_preempts": True, "watchdog_seconds": 3, "physical_test_mode": False},
            }

    def _unavailable(self, rig: RigState, detail: str) -> Dict:
        rig.enabled = False
        rig.state = "Unavailable"
        rig.detail = detail
        rig.updated_at = time.time()
        return rig.payload()

    def start_raydar(self, payload: Dict) -> Dict:
        with self.lock:
            if not payload.get("camera_id"):
                return self._unavailable(self.raydar, "Raydar camera association is not configured")
            url = self.control_url("lighthouse_pose", node_id=payload.get("node_id"), tank_id=payload.get("tank_id"))
            if not url:
                return self._unavailable(self.raydar, "Raydar pose URL is not advertised")
            self.raydar = RigState("Survey", True, "12-point step-and-dwell survey", payload.get("tank_id"), payload.get("camera_id"), time.time())
            self.waypoint_index = 0
            self.tracking_started_at = None
            self.centered_since = None
            self._auto_captured_for_track = False
            self._stop_event.clear()
            survey_url = self.control_url(
                "lighthouse_survey_start",
                node_id=payload.get("node_id"),
                tank_id=payload.get("tank_id"),
            )
            if survey_url:
                try:
                    self.post_json(survey_url, payload)
                    self.raydar.last_command_at = time.time()
                except (urlerror.URLError, TimeoutError, OSError, ValueError):
                    return self._unavailable(self.raydar, "Raydar rejected or lost the survey command")
            elif self.frame_source and (not self._worker or not self._worker.is_alive()):
                self._worker = threading.Thread(target=self._raydar_loop, name="sync-raydar", daemon=True)
                self._worker.start()
            return self.raydar.payload()

    def stop_raydar(self, payload: Dict) -> Dict:
        with self.lock:
            # Disable locally before any network stop, so a failed URL cannot resume commands.
            self.raydar.enabled = False
            self.raydar.state = "STOP"
            self.raydar.detail = payload.get("reason", "Stopped by Sync")
            self.raydar.updated_at = time.time()
            self._stop_event.set()
            url = self.control_url(
                "lighthouse_survey_stop",
                node_id=payload.get("node_id"),
                tank_id=payload.get("tank_id"),
            )
            if url:
                try:
                    self.post_json(url, payload)
                    self.raydar.last_command_at = time.time()
                except (urlerror.URLError, TimeoutError, OSError, ValueError):
                    self.raydar.detail = "Stopped locally; tank stop URL was unreachable"
            return self.raydar.payload()

    def _raydar_loop(self) -> None:
        """Run at 5 FPS during analysis and dwell between survey commands."""
        missing_frames = 0
        next_survey_at = 0.0
        while not self._stop_event.wait(0.2):
            with self.lock:
                if not self.raydar.enabled:
                    return
                camera_id = self.raydar.camera_id
                moving = bool(self.raydar.last_command_at and time.time() - self.raydar.last_command_at < 0.6)
            candidates = []
            if camera_id and self.frame_source and not moving:
                try:
                    jpeg = self.frame_source(camera_id)
                    if jpeg:
                        candidates = self.analyzer.analyze_jpeg(jpeg)
                        missing_frames = 0
                    else:
                        missing_frames += 1
                except (urlerror.URLError, TimeoutError, OSError, ValueError):
                    missing_frames += 1
                if missing_frames >= 15:
                    with self.lock:
                        self._unavailable(self.raydar, "Stale or disconnected frames stopped autonomy")
                    return
            now = time.time()
            if candidates or now >= next_survey_at:
                self.tick_raydar(candidates, now)
                next_survey_at = now + (0.2 if candidates else 2.0)

    def set_manual(self, rig_name: str) -> Dict:
        with self.lock:
            rig = self.raydar if rig_name == "raydar" else self.reeflex
            rig.enabled = False
            rig.state = "Manual"
            rig.detail = "Autonomy preempted by manual controls"
            rig.updated_at = time.time()
            return rig.payload()

    def start_reeflex(self, payload: Dict) -> Dict:
        with self.lock:
            if not payload.get("camera_id"):
                return self._unavailable(self.reeflex, "Reeflex camera association is not configured")
            url = self.control_url("reeflex_idle_start", "reeflex_idle", node_id=payload.get("node_id"), tank_id=payload.get("tank_id"))
            if not url:
                return self._unavailable(self.reeflex, "Reeflex idle-start URL is not advertised")
            self.reeflex = RigState("Survey", True, "Conservative advertised small-arc survey", payload.get("tank_id"), payload.get("camera_id"), time.time())
            try:
                self.post_json(url, payload)
                self.reeflex.last_command_at = time.time()
            except (urlerror.URLError, TimeoutError, OSError, ValueError):
                return self._unavailable(self.reeflex, "Reeflex rejected or lost the start command")
            return self.reeflex.payload()

    def stop_reeflex(self, payload: Dict) -> Dict:
        with self.lock:
            self.reeflex.enabled = False
            self.reeflex.state = "STOP"
            self.reeflex.detail = payload.get("reason", "Stopped by Sync")
            self.reeflex.updated_at = time.time()
            url = self.control_url("reeflex_idle_stop", "reeflex_stop", node_id=payload.get("node_id"), tank_id=payload.get("tank_id"))
            if url:
                try:
                    self.post_json(url, payload)
                    self.reeflex.last_command_at = time.time()
                except (urlerror.URLError, TimeoutError, OSError, ValueError):
                    self.reeflex.detail = "Stopped locally; tank stop URL was unreachable"
            return self.reeflex.payload()

    def tick_raydar(self, candidates: Iterable[Dict], now: Optional[float] = None) -> Dict:
        """Advance one analysis cycle; never sends after a safety failure."""
        now = time.time() if now is None else now
        with self.lock:
            if not self.raydar.enabled:
                return self.raydar.payload()
            url = self.control_url("lighthouse_pose", tank_id=self.raydar.tank_id)
            if not url:
                return self._unavailable(self.raydar, "Raydar URL disappeared")
            target = self.tracker.update(candidates, now)
            if target and target["persistent"]:
                self.last_target = {"x": target["x"], "y": target["y"]}
                if self.raydar.state != "Track":
                    self.tracking_started_at = now
                    self._auto_captured_for_track = False
                self.raydar.state = "Track"
                pan, tilt, centered = proportional_centering((target["x"], target["y"]), self.pose, self.limits)
                self.centered_since = self.centered_since or now if centered else None
                payload = {"pan": pan, "tilt": tilt, "tank_id": self.raydar.tank_id, "source": "sync-autonomy"}
            elif self.raydar.state == "Track" and (self.tracker.lost_for(now) >= 2 or now - (self.tracking_started_at or now) >= 20):
                self.raydar.state = "Survey"
                self.centered_since = None
                self.last_target = None
                point = self.waypoints[self.waypoint_index % len(self.waypoints)]
                self.waypoint_index += 1
                payload = {**point, "tank_id": self.raydar.tank_id, "source": "sync-autonomy"}
            elif self.raydar.state == "Survey":
                point = self.waypoints[self.waypoint_index % len(self.waypoints)]
                self.waypoint_index += 1
                payload = {**point, "tank_id": self.raydar.tank_id, "source": "sync-autonomy"}
            else:
                return self.raydar.payload()
            try:
                self.post_json(url, payload)
            except (urlerror.URLError, TimeoutError, OSError, ValueError):
                return self._unavailable(self.raydar, "Command failed; watchdog stopped autonomy")
            self.pose = (float(payload["pan"]), float(payload["tilt"]))
            self.raydar.last_command_at = now
            self.raydar.updated_at = now
            self.raydar.detail = "Target centered" if self.centered_since else ("Tracking interesting motion" if self.raydar.state == "Track" else "Survey dwell")
            if (self.centered_since and now - self.centered_since >= 1 and not self._auto_captured_for_track
                    and self.capture_callback and self.raydar.camera_id):
                try:
                    self.capture_callback({"camera_id": self.raydar.camera_id, "tank_id": self.raydar.tank_id,
                                           "trigger": "raydar-auto", "label": "Unknown",
                                           "focus_region": {"x": target["x"], "y": target["y"]},
                                           "scores": {"persistence_frames": self.tracker.frames, "centered_seconds": now - self.centered_since}})
                    self._auto_captured_for_track = True
                except (ValueError, OSError):
                    pass
            return self.raydar.payload()


def ask_the_deep(image: bytes, api_key: Optional[str] = None, model: Optional[str] = None,
                 persona: Optional[str] = None, transport: Optional[Callable[[str, Dict, Dict], Dict]] = None) -> Dict:
    """Make exactly one explicit Responses API request. No retry occurs here."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Ask the Deep is disabled: OPENAI_API_KEY is not configured")
    if not is_jpeg(image):
        raise ValueError("Sighting image is not a valid JPEG")
    persona = persona or random.choice(("researcher", "pirate", "captain"))
    model = model or os.environ.get("OPENAI_VISION_MODEL", "gpt-5.6-terra")
    prompt = (
        "Study this aquarium sighting using only visible evidence. Return concise JSON with keys "
        "visual_evidence, possible_subject, uncertainty, interesting_fact, narration. Do not claim a "
        f"confirmed identification. Narration should sound like a curious {persona}, while facts stay cautious."
    )
    body = {
        "model": model,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii"), "detail": "low"},
        ]}],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if transport is None:
        def transport(url: str, request_body: Dict, request_headers: Dict) -> Dict:
            req = urlrequest.Request(url, data=json.dumps(request_body).encode("utf-8"), headers=request_headers, method="POST")
            with urlrequest.urlopen(req, timeout=30) as response:
                return json.loads(response.read(MAX_IMAGE_BYTES))
    response = transport("https://api.openai.com/v1/responses", body, headers)
    output_text = response.get("output_text", "")
    if not output_text:
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_text += content.get("text", "")
    try:
        parsed = json.loads(output_text)
    except (TypeError, json.JSONDecodeError):
        parsed = {"visual_evidence": output_text or "No field note returned", "possible_subject": "Unknown", "uncertainty": "High", "interesting_fact": "", "narration": ""}
    return {"label": AI_LABEL, "persona": persona, "model": model, **parsed}
