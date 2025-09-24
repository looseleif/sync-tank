#!/usr/bin/env python3
import io
import sys
import time
import threading
from typing import Optional

from flask import Flask, Response, render_template_string

from picamera2 import Picamera2
import numpy as np
from PIL import Image

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
  <head>
    <title>Dual Camera Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
      body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
      header { padding: 0.75rem 1rem; background: #222; position: sticky; top: 0; }
      .grid { display: grid; grid-template-columns: 1fr; gap: 12px; padding: 12px; }
      @media (min-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } }
      .card { background: #000; border-radius: 10px; overflow: hidden; }
      .card h2 { margin: 0; padding: 8px 12px; background: #333; font-weight: 600; font-size: 14px; }
      .card img { width: 100%; display: block; background: #000; }
      .hint { padding: 0 12px 12px; color: #aaa; font-size: 12px; }
      a { color: #74c0ff; text-decoration: none; }
    </style>
  </head>
  <body>
    <header>
      <div><strong>Dual Camera Stream</strong> — Pi listening on {{host}}:{{port}}</div>
    </header>
    <div class="grid">
      <div class="card">
        <h2>Camera 0</h2>
        <img src="{{ url_for('stream_cam0') }}" />
      </div>
      <div class="card">
        <h2>Camera 1</h2>
        <img src="{{ url_for('stream_cam1') }}" />
      </div>
    </div>
    <div class="hint">
      If a feed is blank, the sensor may be missing or busy. Ctrl+C and rerun.
    </div>
  </body>
</html>
"""

class FrameBuffer:
    """Stores latest JPEG frame, with a condition for MJPEG generators."""
    def __init__(self):
        self._frame: Optional[bytes] = None
        self._cond = threading.Condition()

    def set(self, jpeg_bytes: bytes):
        with self._cond:
            self._frame = jpeg_bytes
            self._cond.notify_all()

    def get(self) -> bytes:
        with self._cond:
            while self._frame is None:
                self._cond.wait()
            return self._frame

def encode_jpeg_from_array(arr: np.ndarray, quality: int = 80) -> bytes:
    # arr is in RGB888
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def camera_loop(cam_index: int, fb: FrameBuffer, width=1280, height=720, fps=20):
    cam = None
    try:
        cam = Picamera2(camera_num=cam_index)
        config = cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        cam.configure(config)
        cam.start()
        # Warm-up
        time.sleep(0.2)
        period = 1.0 / float(max(1, fps))
        while True:
            t0 = time.time()
            frame = cam.capture_array("main")  # RGB888 numpy array
            jpeg = encode_jpeg_from_array(frame, quality=80)
            fb.set(jpeg)
            dt = time.time() - t0
            # simple pacing
            if dt < period:
                time.sleep(period - dt)
    except Exception as e:
        print(f"[ERROR] Camera {cam_index} loop exited: {e}", file=sys.stderr)
    finally:
        try:
            if cam:
                cam.stop()
                cam.close()
        except Exception:
            pass

def mjpeg_generator(fb: FrameBuffer):
    boundary = b"--frame"
    while True:
        frame = fb.get()
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
            frame + b"\r\n"
        )

# Globals
framebuffers: list[FrameBuffer] = []
threads: list[threading.Thread] = []

@app.route("/")
def index():
    host = "0.0.0.0"
    port = app.config.get("PORT", 8000)
    return render_template_string(HTML, host=host, port=port)

@app.route("/stream0")
def stream_cam0():
    if len(framebuffers) < 1:
        return "Camera 0 not available", 503
    return Response(mjpeg_generator(framebuffers[0]),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream1")
def stream_cam1():
    if len(framebuffers) < 2:
        return "Camera 1 not available", 503
    return Response(mjpeg_generator(framebuffers[1]),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def main():
    # Start up to two cameras
    started = 0
    for idx in (0, 1):
        fb = FrameBuffer()
        try:
            th = threading.Thread(target=camera_loop, args=(idx, fb), daemon=True)
            th.start()
            framebuffers.append(fb)
            threads.append(th)
            started += 1
        except Exception as e:
            print(f"[WARN] Could not start camera {idx}: {e}", file=sys.stderr)

    if started == 0:
        print("[ERROR] No cameras started. Check connections and overlays.", file=sys.stderr)
        sys.exit(1)

    app.config["PORT"] = 8000
    app.run(host="0.0.0.0", port=8000, threaded=True)

if __name__ == "__main__":
    main()
