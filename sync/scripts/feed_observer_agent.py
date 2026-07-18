#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib import error, request
from urllib.parse import urljoin


def get_json(url: str, timeout_seconds: float = 3.0) -> dict | None:
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def post_json(url: str, payload: dict, timeout_seconds: float = 3.0) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response.read()
            return 200 <= response.status < 300
    except error.URLError:
        return False


def fetch_bytes(url: str, timeout_seconds: float = 3.0, max_bytes: int = 256_000) -> bytes | None:
    try:
        with request.urlopen(url, timeout=timeout_seconds) as response:
            return response.read(max_bytes)
    except (error.URLError, TimeoutError):
        return None


def extract_jpeg_frame(data: bytes | None) -> bytes | None:
    if not data:
        return None
    start = data.find(b"\xff\xd8")
    if start < 0:
        return data
    end = data.find(b"\xff\xd9", start + 2)
    if end < 0:
        return data[start:]
    return data[start : end + 2]


def camera_snapshot_url(camera: dict) -> str | None:
    return camera.get("snapshot_url") or camera.get("latest_image_url")


def camera_sample_urls(camera: dict) -> list[str]:
    urls = []
    for key in ("snapshot_url", "latest_image_url", "stream_url"):
        url = camera.get(key)
        if url and url not in urls:
            urls.append(url)
    return urls


def signature(data: bytes, buckets: int = 64) -> list[int]:
    if not data:
        return []
    step = max(1, len(data) // buckets)
    values = []
    for offset in range(0, min(len(data), step * buckets), step):
        chunk = data[offset : offset + step]
        values.append(sum(chunk) // max(1, len(chunk)))
    return values


def motion_score(previous: list[int] | None, current: list[int]) -> float:
    if not previous or not current:
        return 0.0
    count = min(len(previous), len(current))
    if count == 0:
        return 0.0
    return sum(abs(current[index] - previous[index]) for index in range(count)) / (count * 255)


def focus_region(previous: list[int] | None, current: list[int], grid_size: int = 8) -> dict:
    if not previous or not current:
        return {"x": 0.5, "y": 0.5, "width": 0.24, "height": 0.24, "confidence": 0.0}
    count = min(len(previous), len(current), grid_size * grid_size)
    if count == 0:
        return {"x": 0.5, "y": 0.5, "width": 0.24, "height": 0.24, "confidence": 0.0}
    deltas = [abs(current[index] - previous[index]) for index in range(count)]
    strongest = max(range(count), key=lambda index: deltas[index])
    row = strongest // grid_size
    column = strongest % grid_size
    confidence = min(1.0, deltas[strongest] / 255)
    return {
        "x": (column + 0.5) / grid_size,
        "y": (row + 0.5) / grid_size,
        "width": 1 / grid_size * 1.7,
        "height": 1 / grid_size * 1.7,
        "confidence": round(confidence, 4),
        "method": "byte_signature_grid",
    }


def fish_candidate_score(camera: dict, motion: float, focus: dict) -> float:
    camera_type = f"{camera.get('camera_type', '')} {camera.get('item_type', '')} {camera.get('source_type', '')}".lower()
    is_visual_camera = any(token in camera_type for token in ("scope", "camera", "usb", "lighthouse", "reeflex", "floater"))
    if not is_visual_camera:
        return 0.0
    focus_confidence = float(focus.get("confidence") or 0)
    score = (motion * 5.5) + (focus_confidence * 0.45)
    return round(max(0.0, min(1.0, score)), 4)


class FeedObserver:
    def __init__(
        self,
        organizer_base: str,
        storage_dir: str,
        threshold: float,
        burst_frames: int,
        deep_link_base: str | None = None,
        display_base: str | None = None,
    ) -> None:
        self.organizer_base = organizer_base.rstrip("/")
        self.deep_link_base = deep_link_base.rstrip("/") if deep_link_base else None
        self.display_base = (display_base or organizer_base).rstrip("/")
        self.event_dir = Path(storage_dir) / "observer_events"
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.burst_frames = burst_frames
        self.previous: dict[str, list[int]] = {}
        self.last_event_at: dict[str, float] = {}

    def observe_once(self) -> int:
        layout = get_json(f"{self.organizer_base}/api/layout") or {}
        cameras = [camera for camera in layout.get("cameras", []) if camera.get("status") == "online"]
        events = 0
        for camera in cameras:
            if self.observe_camera(camera):
                events += 1
        return events

    def observe_camera(self, camera: dict) -> bool:
        camera_id = camera.get("camera_id") or camera.get("id")
        urls = camera_sample_urls(camera)
        if not camera_id or not urls:
            return False
        data = None
        for url in urls:
            data = extract_jpeg_frame(fetch_bytes(url))
            if data:
                break
        if not data:
            return False
        current = signature(data)
        previous = self.previous.get(camera_id)
        score = motion_score(previous, current)
        focus = focus_region(previous, current)
        self.previous[camera_id] = current
        now = time.time()
        if score < self.threshold or now - self.last_event_at.get(camera_id, 0) < 8:
            return False
        self.last_event_at[camera_id] = now
        event_id = f"{camera_id}-{int(now)}-{hashlib.sha1(data[:2048]).hexdigest()[:8]}"
        frame_urls = self.capture_burst(camera, event_id, data)
        fish_score = fish_candidate_score(camera, score, focus)
        classifier_label = "fish_candidate" if fish_score >= 0.42 else "motion"
        activity_score = round(min(1.0, (score * 8) + (fish_score * 0.22)), 4)
        payload = {
            "observation_id": event_id,
            "camera_id": camera_id,
            "tank_id": camera.get("tank_id", "main-tank"),
            "node_id": camera.get("node_id") or camera.get("hub_id"),
            "event_type": "motion",
            "motion_score": round(score, 4),
            "classifier_label": classifier_label,
            "classifier_confidence": fish_score,
            "fish_candidate_score": fish_score,
            "activity_score": activity_score,
            "source_processor": "pi_index_observer",
            "focus_point": {"x": focus["x"], "y": focus["y"], "confidence": focus["confidence"]},
            "focus_region": focus,
            "frame_urls": frame_urls,
            "summary": "Possible fish or moving organism." if classifier_label == "fish_candidate" else "Motion observed. Identity unknown.",
            "identity_status": "needs_label",
            "classifier_status": "queued_for_pc" if self.deep_link_base else "pi_heuristic_ready",
            "created_at": now,
        }
        organizer_ok = post_json(f"{self.organizer_base}/api/observations/register", payload)
        deep_link_ok = self.forward_to_deep_link(payload, camera)
        if self.deep_link_base:
            status_update = {
                **payload,
                "classifier_status": "pc_forwarded" if deep_link_ok else "pc_unavailable",
            }
            post_json(f"{self.organizer_base}/api/observations/register", status_update)
        return organizer_ok or deep_link_ok

    def forward_to_deep_link(self, observation: dict, camera: dict) -> bool:
        if not self.deep_link_base:
            return False
        frame_urls = [
            url if url.startswith("http://") or url.startswith("https://") else urljoin(f"{self.display_base}/", url.lstrip("/"))
            for url in observation.get("frame_urls", [])
        ]
        payload = {
            **observation,
            "frame_urls": frame_urls,
            "display_observation_url": f"{self.display_base}/api/layout",
            "source_camera": {
                "camera_id": camera.get("camera_id") or camera.get("id"),
                "label": camera.get("label") or camera.get("name"),
                "camera_type": camera.get("camera_type"),
                "source_type": camera.get("source_type"),
                "stream_url": camera.get("stream_url"),
                "snapshot_url": camera.get("snapshot_url") or camera.get("latest_image_url"),
            },
            "classifier_request": {
                "requested": True,
                "task": "identify_visible_organism_or_motion_source",
                "input_type": "frame_burst_urls",
                "focus_region": observation.get("focus_region"),
            },
        }
        endpoints = (
            "/api/observations/register",
            "/api/classifier/observations/register",
            "/api/sync-tank/observations/register",
        )
        for endpoint in endpoints:
            if post_json(f"{self.deep_link_base}{endpoint}", payload):
                return True
        return False

    def capture_burst(self, camera: dict, event_id: str, first_frame: bytes) -> list[str]:
        camera_id = camera.get("camera_id") or camera.get("id")
        urls = camera_sample_urls(camera)
        target_dir = self.event_dir / event_id
        target_dir.mkdir(parents=True, exist_ok=True)
        frames = [first_frame]
        for _ in range(max(0, self.burst_frames - 1)):
            time.sleep(0.35)
            data = None
            for url in urls:
                data = extract_jpeg_frame(fetch_bytes(url))
                if data:
                    break
            if data:
                frames.append(data)
        frame_urls = []
        for index, data in enumerate(frames):
            path = target_dir / f"frame-{index + 1}.jpg"
            path.write_bytes(data)
            frame_urls.append(f"/observer_events/{event_id}/frame-{index + 1}.jpg")
        return frame_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observe Sync Tank camera feeds for motion events")
    parser.add_argument("--organizer-base", default="http://127.0.0.1:8765")
    parser.add_argument("--storage-dir", default="/home/zero/sync-tank/storage")
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.045)
    parser.add_argument("--burst-frames", type=int, default=5)
    parser.add_argument("--deep-link-base", default=None, help="optional main PC backend URL to receive motion/classifier events")
    parser.add_argument("--home-server-base", default=None, help="alias for --deep-link-base")
    parser.add_argument("--processing-node-base", default=None, help="alias for --deep-link-base for the processing/classifier computer")
    parser.add_argument("--display-base", default=None, help="display node URL visible to the main PC for observer frame URLs")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observer = FeedObserver(
        args.organizer_base,
        args.storage_dir,
        args.threshold,
        args.burst_frames,
        deep_link_base=args.deep_link_base or args.home_server_base or args.processing_node_base,
        display_base=args.display_base,
    )
    while True:
        observer.observe_once()
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
