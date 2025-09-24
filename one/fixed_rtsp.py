#!/usr/bin/env python3
import os
import sys
import time
import threading
import signal
import subprocess
from typing import Optional, List, Dict, Tuple
from flask import Flask, Response, render_template_string, jsonify
import cv2

print("[BOOT] fixed_rtsp_flask starting...", flush=True)

# Force OpenCV/FFmpeg to use TCP for RTSP unless we override for a cam
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# ------------------ CONFIG ------------------
PORT = int(os.environ.get("PORT", "8002"))

# Two known-good feeds you had working earlier:
KNOWN_URLS = [
    "rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/101",
    "rtsp://admin:admin@192.168.0.108:554/h264/ch1/main/av_stream",
]

# Optional third-cam hunter target (non-blocking background probe)
# Example: HUNT_TARGET=192.168.0.108:554
HUNT_TARGET = os.environ.get("HUNT_TARGET", "").strip()

# For hunting, try these common paths (kept tight so we don’t waste time)
HUNT_PATHS = [
    "/h264/ch1/main/av_stream",            # Dahua/Amcrest-ish
    "/live/ch00_0", "/live", "/videoMain", # many white-labels
    "/Streaming/Channels/101",             # Hikvision-ish
    "/h264Preview_01_main",                # Reolink-ish
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
]

# Try these creds combinations on the hunter
HUNT_CREDS = [
    ("admin", "admin"),
    ("admin", "123456"),
    ("admin", ""),        # blank password
]

# Streaming params
FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

# FFmpeg reconnect flags passed through OpenCV (helps with flakey links)
# These only work on newer ffmpeg builds; harmless if ignored.
FFMPEG_OPEN_FLAGS = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|rw_timeout;2000000|stimeout;2000000"
)

# --------------- APP STATE ------------------
app = Flask(__name__)

class FrameBuffer:
    def __init__(self):
        self._buf: Optional[bytes] = None
        self._cond = threading.Condition()

    def set(self, b: bytes):
        with self._cond:
            self._buf = b
            self._cond.notify_all()

    def get(self) -> bytes:
        with self._cond:
            while self._buf is None:
                self._cond.wait()
            return self._buf

class CamWorker(threading.Thread):
    def __init__(self, idx: int, url: str, title: Optional[str] = None):
        super().__init__(daemon=True)
        self.idx = idx
        self.url = url
        self.title = title or url
        self.buf = FrameBuffer()
        self._stop = threading.Event()
        self._open_flags = FFMPEG_OPEN_FLAGS

    def stop(self):
        self._stop.set()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        # Try TCP first (env default). If a URL needs UDP, you can set
        # OPENCV_FFMPEG_CAPTURE_OPTIONS per-process or duplicate worker logic.
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        return None

    def run(self):
        print(f"[CAM{self.idx}] starting worker for {self.url}", flush=True)
        last_ok = 0.0
        while not self._stop.is_set():
            cap = self._open_capture()
            if not cap:
                print(f"[CAM{self.idx}] open failed; retry in 2s", flush=True)
                time.sleep(2)
                continue

            # Set props (not all cameras honor these)
            if FRAME_W: cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            if FRAME_H: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            if FRAME_FPS: cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"[CAM{self.idx}] read error; reopening in 1s", flush=True)
                    cap.release()
                    time.sleep(1)
                    break

                # Resize only if frame size wildly off (avoid extra CPU if not needed)
                if FRAME_W and FRAME_H:
                    h, w = frame.shape[:2]
                    if abs(w - FRAME_W) > 8 or abs(h - FRAME_H) > 8:
                        frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

                ok, jpg = cv2.imencode(".jpg", frame, encode_params)
                if ok:
                    self.buf.set(jpg.tobytes())
                    last_ok = time.time()
                else:
                    print(f"[CAM{self.idx}] encode fail; continuing", flush=True)

            # loop back to reopen
        print(f"[CAM{self.idx}] stopped", flush=True)

class CamRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cams: Dict[int, CamWorker] = {}
        self._next_idx = 0

    def add(self, url: str, title: Optional[str] = None) -> int:
        with self._lock:
            idx = self._next_idx
            self._next_idx += 1
            worker = CamWorker(idx, url, title)
            self._cams[idx] = worker
            worker.start()
            print(f"[REG] added cam {idx}: {url}", flush=True)
            return idx

    def list(self) -> List[Tuple[int, str, str]]:
        with self._lock:
            return [(i, c.url, c.title) for i, c in sorted(self._cams.items())]

    def get(self, idx: int) -> Optional[CamWorker]:
        with self._lock:
            return self._cams.get(idx)

    def has_url(self, url: str) -> bool:
        with self._lock:
            return any(c.url == url for c in self._cams.values())

    def stop_all(self):
        with self._lock:
            for c in self._cams.values():
                c.stop()

cams = CamRegistry()

# ----------------- HUNTER --------------------
def ffprobe_video(url: str, transport: str = "tcp", timeout_s: float = 3.0) -> bool:
    """
    Quick check for a video stream using ffprobe.
    Returns True if a video stream is detected.
    """
    args = [
        "ffprobe",
        "-v", "error",
        "-hide_banner",
        "-rtsp_transport", transport,
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=nokey=1:noprint_wrappers=1",
        url,
    ]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        if out.returncode == 0 and "video" in out.stdout:
            return True
        return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def build_urls(hostport: str) -> List[str]:
    host, _, port = hostport.partition(":")
    port = port or "554"
    urls: List[str] = []
    # Try user:pass@host format
    for u, p in HUNT_CREDS:
        for path in HUNT_PATHS:
            urls.append(f"rtsp://{u}:{p}@{host}:{port}{path}")
    # Also try path without credentials (some cams allow anon)
    for path in HUNT_PATHS:
        urls.append(f"rtsp://{host}:{port}{path}")
    return urls

def hunter_thread(hostport: str):
    print(f"[HUNT] starting background hunt for {hostport}", flush=True)
    tried = 0
    for url in build_urls(hostport):
        tried += 1
        # Try UDP first for those that dislike TCP
        if ffprobe_video(url, "udp", 2.5) or ffprobe_video(url, "tcp", 2.5):
            if not cams.has_url(url):
                cams.add(url, f"Hunted @ {hostport}")
            print(f"[HUNT] FOUND video at {url}", flush=True)
            return
        if tried % 10 == 0:
            print(f"[HUNT] tried {tried} candidates so far...", flush=True)
    print(f"[HUNT] done; nothing found for {hostport}", flush=True)

# --------------- FLASK VIEWS ----------------
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <title>RTSP Feeds — MJPEG</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
      :root { color-scheme: dark; }
      body { font-family: system-ui, sans-serif; margin: 0; background: #0b0b0b; color: #eee; }
      header { padding: .75rem 1rem; background: #161616; border-bottom: 1px solid #222; position: sticky; top: 0; }
      .grid { display: grid; gap: 12px; padding: 12px; grid-template-columns: 1fr; }
      @media (min-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } }
      @media (min-width: 1400px) { .grid { grid-template-columns: 1fr 1fr 1fr; } }
      .card { background: #000; border: 1px solid #1f1f1f; border-radius: 10px; overflow: hidden; }
      .card h2 { margin: 0; padding: 8px 12px; background: #222; font-size: 14px; border-bottom: 1px solid #1f1f1f; }
      .card .sub { color: #9aa; font-size: 12px; padding: 8px 12px; }
      .card img { width: 100%; display: block; background: #000; aspect-ratio: 16/9; object-fit: contain; }
      .hint { padding: 8px 12px 16px; color: #9aa; font-size: 12px; }
      code { background: #161616; padding: 2px 6px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <header><strong>RTSP Feeds — MJPEG</strong> — http://{{host}}:{{port}}</header>
    <div class="grid">
      {% for cam in cams %}
      <div class="card">
        <h2>{{ cam.title }}</h2>
        <div class="sub">{{ cam.url }}</div>
        <img src="{{ url_for('stream_cam', idx=cam.idx) }}" />
      </div>
      {% endfor %}
    </div>
    <div class="hint">
      Endpoints: {% for cam in cams %}/cam/{{cam.idx}} {% endfor %} — health: /health — add: POST /add (json: {"url": "rtsp://..."})
    </div>
  </body>
</html>
"""

@app.route("/")
def index():
    items = [{"idx": i, "url": u, "title": t} for i, u, t in cams.list()]
    host = os.environ.get("HOST", "0.0.0.0")
    return render_template_string(INDEX_HTML, cams=items, host=host, port=PORT)

@app.route("/cam/<int:idx>")
def stream_cam(idx: int):
    worker = cams.get(idx)
    if worker is None:
        return "Camera not found", 404

    def gen():
        boundary = b"--frame\r\n"
        while True:
            frame = worker.buf.get()
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    return jsonify({
        "cams": [{"idx": i, "url": u, "title": t} for i, u, t in cams.list()],
        "status": "ok"
    })

from flask import request
@app.route("/add", methods=["POST"])
def add_cam():
    j = request.get_json(force=True, silent=True) or {}
    url = j.get("url")
    title = j.get("title")
    if not url:
        return jsonify({"ok": False, "error": "missing url"}), 400
    if cams.has_url(url):
        return jsonify({"ok": True, "idx": None, "msg": "already exists"})
    idx = cams.add(url, title)
    return jsonify({"ok": True, "idx": idx})

# --------------- MAIN -----------------------
def start_known():
    for url in KNOWN_URLS:
        cams.add(url, f"Fixed: {url}")

def start_hunter_if_any():
    if HUNT_TARGET:
        t = threading.Thread(target=hunter_thread, args=(HUNT_TARGET,), daemon=True)
        t.start()
        print(f"[MAIN] hunter launched for {HUNT_TARGET}", flush=True)
    else:
        print("[MAIN] no hunter target configured (set HUNT_TARGET=ip:port to enable).", flush=True)

def handle_sigint(sig, frame):
    print("\n[MAIN] SIGINT received, stopping...", flush=True)
    cams.stop_all()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, handle_sigint)
    start_known()
    start_hunter_if_any()
    host = "0.0.0.0"
    print(f"[MAIN] serving on http://{host}:{PORT}/", flush=True)
    app.run(host=host, port=PORT, threaded=True)

if __name__ == "__main__":
    main()
