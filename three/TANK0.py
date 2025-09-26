#!/usr/bin/env python3
# csi_flask_simple.py — Headless Flask MJPEG from Raspberry Pi 5 CSI cams (Picamera2)
# - Enumerates cameras dynamically
# - Background thread per cam: capture -> Pillow JPEG -> serve latest
# - No Qt / no libcamera-apps required

import io, time, threading
from typing import Dict, Optional, List
from flask import Flask, Response, abort, jsonify, render_template_string
from picamera2 import Picamera2
from PIL import Image

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<meta charset="utf-8">
<title>Pi 5 CSI Streams</title>
<style>
  :root{color-scheme:light dark}
  body{font-family:system-ui,Arial,sans-serif;margin:16px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}
  img{width:100%;height:auto;background:#000;border-radius:12px}
  .card{border:1px solid #ddd;border-radius:14px;padding:12px}
  h1{margin:0 0 .75rem 0;font-size:1.25rem}
  h2{margin:0 0 .5rem 0;font-size:1rem}
  code{background:#f5f5f5;padding:2px 6px;border-radius:6px}
  .muted{opacity:.7}
</style>
<h1>Raspberry Pi 5 CSI Streams (headless)</h1>
<p class="muted">Health: <code>/health</code> • Streams: <code>/csi/&lt;idx&gt;.mjpg</code></p>
{% if cams %}
<div class="grid">
  {% for idx,desc in cams %}
  <div class="card">
    <h2>Cam {{idx}} — <code>/csi/{{idx}}.mjpg</code></h2>
    <div class="muted" style="margin-bottom:.5rem">{{desc}}</div>
    <a href="/csi/{{idx}}.mjpg"><img src="/csi/{{idx}}.mjpg" alt="cam {{idx}}"></a>
  </div>
  {% endfor %}
</div>
{% else %}
<p>No cameras detected by Picamera2.</p>
{% endif %}
"""

class SimpleCam:
    """Capture loop that keeps the latest JPEG for MJPEG streaming."""
    def __init__(self, cam_index:int, size=(1280,720), fps:int=15):
        self.idx = cam_index
        self.size = size
        self.fps  = max(1, min(int(fps), 30))
        self.picam: Optional[Picamera2] = None
        self.latest_jpeg: Optional[bytes] = None
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._t and self._t.is_alive():
            return
        self.picam = Picamera2(camera_num=self.idx)
        cfg = self.picam.create_preview_configuration(main={"size": self.size, "format": "RGB888"})
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(0.3)  # quick AE/AWB warmup
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        if self.picam:
            try: self.picam.stop()
            except Exception: pass
            try: self.picam.close()
            except Exception: pass
        self.picam = None
        self._t = None

    def _loop(self):
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                arr = self.picam.capture_array("main")  # numpy array (H,W,3) RGB
                with io.BytesIO() as buff:
                    Image.fromarray(arr).save(buff, format="JPEG", quality=70, optimize=True)
                    self.latest_jpeg = buff.getvalue()
            except Exception:
                # keep last good frame and continue
                pass
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def frames(self):
        boundary = b"--frame\r\n"
        while True:
            if self.latest_jpeg is None:
                time.sleep(0.05)
                continue
            jpg = self.latest_jpeg
            yield (boundary +
                   b"Content-Type: image/jpeg\r\n" +
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")

_streams: Dict[int, SimpleCam] = {}
_detected: List[str] = []

def _detect_cams():
    global _detected
    try:
        info = Picamera2.global_camera_info()
        _detected = [str(i) for i in info]
        return list(range(len(info)))
    except Exception:
        _detected = []
        return []

def _get_stream(idx:int) -> SimpleCam:
    if idx not in _streams:
        cam = SimpleCam(idx, size=(1280,720), fps=15)
        cam.start()
        _streams[idx] = cam
    return _streams[idx]

@app.route("/")
def index():
    idxs = _detect_cams()
    cams = [(i, _detected[i] if i < len(_detected) else "") for i in idxs]
    return render_template_string(INDEX_HTML, cams=cams)

@app.route("/health")
def health():
    return jsonify(status="ok", cams=_detected), 200

@app.route("/csi/<int:idx>.mjpg")
def csi(idx:int):
    # ensure idx exists (based on last detection)
    idxs = _detect_cams()
    if idx not in idxs:
        abort(404, f"CSI camera index {idx} not available")
    try:
        stream = _get_stream(idx)
    except Exception as e:
        abort(404, f"Failed to open camera {idx}: {e}")
    return Response(stream.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("[BOOT] csi_flask_simple on :8080 (no Qt, headless)")
    app.run(host="0.0.0.0", port=8080, threaded=True)
