#!/usr/bin/env python3
"""
synctankone — Flask preview for CSI + RTSP with bulletproof Ctrl-C shutdown.

Endpoints:
  GET /              — dashboard
  GET /csis          — list CSI indices
  GET /csi/<idx>.mjpg  — MJPEG stream from CSI cam
  GET /csi/<idx>.jpg   — last JPEG from CSI cam
  GET /rtsp           — list RTSP names
  GET /rtsp/<name>.mjpg — MJPEG stream from RTSP cam
  GET /rtsp/<name>.jpg  — last JPEG from RTSP cam

Edit RTSP_SOURCES below to your cameras.
"""

import os, io, time, signal, threading, sys
from typing import Dict, Optional, List, Tuple
from flask import Flask, jsonify, Response, make_response, render_template_string, abort
from werkzeug.serving import make_server

# ------------------------
# Config
# ------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8002"))

CSI_W = int(os.environ.get("CSI_W", "1280"))
CSI_H = int(os.environ.get("CSI_H", "720"))
CSI_FPS = int(os.environ.get("CSI_FPS", "10"))
CSI_JPEG_QUALITY = int(os.environ.get("CSI_JPEG_QUALITY", "70"))

# Put your RTSP cameras here (name -> URL)
RTSP_SOURCES: Dict[str, str] = {
    "ipcam1": os.environ.get("RTSP0", "rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream"),
    "ipcam2": os.environ.get("RTSP1", "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/101"),
    "ipcam3": os.environ.get("RTSP2", "rtsp://admin:admin@192.168.0.108:554/h264/ch1/main/av_stream"),
}

# ------------------------
# Globals
# ------------------------
app = Flask(__name__)
STOP = threading.Event()
WORKERS: List[threading.Thread] = []
PRINT_LOCK = threading.Lock()

def log(msg: str):
    with PRINT_LOCK:
        print(msg, flush=True)

# Try Picamera2
try:
    from picamera2 import Picamera2
    from PIL import Image
    PICAM_AVAILABLE = True
except Exception as e:
    PICAM_AVAILABLE = False
    _IMPORT_ERR = str(e)

# Try OpenCV for RTSP pulls
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception as e:
    OPENCV_AVAILABLE = False
    _CV_ERR = str(e)

# ------------------------
# CSI camera worker
# ------------------------
class CSICam:
    def __init__(self, idx: int, size=(1280, 720), fps: int = 10, q: int = 70):
        self.idx = idx
        self.size = size
        self.fps = max(1, min(fps, 30))
        self.q = max(1, min(q, 95))
        self.picam: Optional['Picamera2'] = None
        self.latest: Optional[bytes] = None
        self.err = ""
        self.ok = False
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if not PICAM_AVAILABLE:
            self.err = f"picamera2 not available: {_IMPORT_ERR}"
            return
        try:
            self.picam = Picamera2(camera_num=self.idx)

            # Request a single RGB stream (avoid RAW path issues)
            cfg = self.picam.create_video_configuration(
                main={"size": self.size, "format": "XRGB8888"},
                raw=None,
                buffer_count=4,
            )
            self.picam.configure(cfg)

            # Lock FPS (optional, stabilizes timings)
            us = int(1_000_000 // self.fps)
            try:
                self.picam.set_controls({"FrameDurationLimits": (us, us)})
            except Exception:
                pass

            self.picam.start()
            time.sleep(0.3)

            self._stop.clear()
            self._t = threading.Thread(target=self._loop, name=f"csi{self.idx}", daemon=True)
            self._t.start()
            self.ok = True
            log(f"[CSI{self.idx}] started")
        except Exception as e:
            self.err = f"start: {e}"
            try:
                if self.picam:
                    self.picam.stop()
            except Exception:
                pass
            try:
                if self.picam:
                    self.picam.close()
            except Exception:
                pass
            self.picam = None

    def _loop(self):
        period = 1.0 / self.fps
        while not STOP.is_set() and not self._stop.is_set():
            t0 = time.time()
            try:
                arr = self.picam.capture_array("main")
                # to JPEG
                im = Image.fromarray(arr)
                with io.BytesIO() as b:
                    im.save(b, format="JPEG", quality=self.q, optimize=True)
                    self.latest = b.getvalue()
            except Exception as e:
                self.err = f"cap: {e}"
            dt = time.time() - t0
            if dt < period:
                # small sleep; exit quickly on STOP
                time.sleep(min(period - dt, 0.05))

    def frames(self):
        boundary = b"--frame\r\n"
        while not STOP.is_set() and not self._stop.is_set():
            if self.latest is None:
                time.sleep(0.05)
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + self.latest + b"\r\n"
            # gentle pacing for MJPEG
            time.sleep(0.15)

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        if self.picam:
            try:
                self.picam.stop()
            except Exception:
                pass
            try:
                self.picam.close()
            except Exception:
                pass
        self.picam = None
        self.ok = False


class CSIRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cams: Dict[int, CSICam] = {}
        self._info: List[str] = []

    def detect(self) -> List[int]:
        if not PICAM_AVAILABLE:
            self._info = []
            return []
        try:
            info = Picamera2.global_camera_info()
            self._info = [str(x) for x in info]
            return list(range(len(info)))
        except Exception:
            self._info = []
            return []

    def start_all(self):
        with self._lock:
            avail = self.detect()
            for i in avail:
                c = CSICam(i, size=(CSI_W, CSI_H), fps=CSI_FPS, q=CSI_JPEG_QUALITY)
                self._cams[i] = c
                c.start()
            if not avail:
                log("[CSI] none detected")

    def list(self) -> List[int]:
        with self._lock:
            return sorted(self._cams.keys())

    def get(self, idx: int) -> Optional[CSICam]:
        with self._lock:
            return self._cams.get(idx)

    def stop_all(self):
        with self._lock:
            for c in self._cams.values():
                c.stop()
            self._cams.clear()


csis = CSIRegistry()

# ------------------------
# RTSP camera worker (OpenCV)
# ------------------------
class RTSPCam:
    def __init__(self, name: str, url: str, fps: int = 10, q: int = 70):
        self.name = name
        self.url = url
        self.fps = max(1, min(fps, 30))
        self.q = max(1, min(q, 95))
        self.cap: Optional['cv2.VideoCapture'] = None
        self.latest: Optional[bytes] = None
        self.err = ""
        self.ok = False
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if not OPENCV_AVAILABLE:
            self.err = f"opencv not available: {_CV_ERR}"
            return
        try:
            # Force TCP transport where supported
            # (OpenCV respects environment variable on some builds)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not self.cap.isOpened():
                self.err = "open failed"
                try:
                    if self.cap:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None
                return

            self._stop.clear()
            self._t = threading.Thread(target=self._loop, name=f"rtsp-{self.name}", daemon=True)
            self._t.start()
            self.ok = True
            log(f"[RTSP:{self.name}] started")
        except Exception as e:
            self.err = f"start: {e}"
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _loop(self):
        period = 1.0 / self.fps
        while not STOP.is_set() and not self._stop.is_set():
            t0 = time.time()
            try:
                ok, frame = self.cap.read()
                if not ok:
                    # brief backoff; let STOP break quickly
                    time.sleep(0.1)
                    continue
                # Encode JPEG
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.q])
                if ok2:
                    self.latest = buf.tobytes()
            except Exception as e:
                self.err = f"cap: {e}"
                time.sleep(0.2)
            dt = time.time() - t0
            if dt < period:
                time.sleep(min(period - dt, 0.05))

    def frames(self):
        boundary = b"--frame\r\n"
        while not STOP.is_set() and not self._stop.is_set():
            if self.latest is None:
                time.sleep(0.05)
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + self.latest + b"\r\n"
            time.sleep(0.15)

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.ok = False


class RTSPRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cams: Dict[str, RTSPCam] = {}

    def start_all(self, sources: Dict[str, str]):
        with self._lock:
            for name, url in sources.items():
                c = RTSPCam(name, url, fps=10, q=75)
                self._cams[name] = c
                c.start()

    def list(self) -> List[str]:
        with self._lock:
            return sorted(self._cams.keys())

    def get(self, name: str) -> Optional[RTSPCam]:
        with self._lock:
            return self._cams.get(name)

    def stop_all(self):
        with self._lock:
            for c in self._cams.values():
                c.stop()
            self._cams.clear()


rtsps = RTSPRegistry()

# ------------------------
# Flask UI
# ------------------------
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>synctankone</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0b0b0b;color:#eee;font-family:system-ui,sans-serif}
header{padding:.6rem 1rem;background:#141414;border-bottom:1px solid #222}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;padding:8px}
.card{background:#000;border:1px solid #222;border-radius:8px;overflow:hidden}
.card img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;display:block}
small{color:#9aa}
h2{font-size:1rem;margin:.5rem 1rem}
</style></head>
<body>
<header><strong>synctankone</strong> <small>MJPEG: /csi/&lt;idx&gt;.mjpg, /rtsp/&lt;name&gt;.mjpg</small></header>
<h2>CSI</h2>
<div class="grid" id="g1"></div>
<h2>RTSP</h2>
<div class="grid" id="g2"></div>
<script>
async function load(){
  let r = await fetch('/csis'); let j = await r.json();
  const g1 = document.getElementById('g1'); g1.innerHTML='';
  (j.items||[]).forEach(c=>{
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<img src="/csi/${c.idx}.mjpg" />`;
    g1.appendChild(d);
  });

  r = await fetch('/rtsp'); j = await r.json();
  const g2 = document.getElementById('g2'); g2.innerHTML='';
  (j.items||[]).forEach(c=>{
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<img src="/rtsp/${c.name}.mjpg" />`;
    g2.appendChild(d);
  });
}
load();
</script>
</body></html>"""

@app.route("/")
def index():
    return PAGE

@app.route("/csis")
def list_csis():
    return jsonify({"items":[{"idx":i} for i in csis.list()]})

@app.route("/rtsp")
def list_rtsp():
    return jsonify({"items":[{"name":n} for n in rtsps.list()]})

@app.route("/csi/<int:idx>.mjpg")
def csi_mjpg(idx:int):
    cam = csis.get(idx)
    if not cam: abort(404)
    return Response(cam.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/csi/<int:idx>.jpg")
def csi_jpg(idx:int):
    cam = csis.get(idx)
    if not cam or cam.latest is None: abort(503, "no frame")
    r = make_response(cam.latest); r.headers["Content-Type"]="image/jpeg"; return r

@app.route("/rtsp/<name>.mjpg")
def rtsp_mjpg(name:str):
    cam = rtsps.get(name)
    if not cam: abort(404)
    return Response(cam.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/rtsp/<name>.jpg")
def rtsp_jpg(name:str):
    cam = rtsps.get(name)
    if not cam or cam.latest is None: abort(503, "no frame")
    r = make_response(cam.latest); r.headers["Content-Type"]="image/jpeg"; return r

# ------------------------
# Flask server (controllable)
# ------------------------
class FlaskServerThread(threading.Thread):
    def __init__(self, flask_app, host, port):
        super().__init__(name="flask", daemon=True)
        self.srv = make_server(host, port, flask_app)
        self.ctx = flask_app.app_context()
        self.ctx.push()
    def run(self):
        self.srv.serve_forever()
    def shutdown(self):
        try: self.srv.shutdown()
        except Exception: pass

# ------------------------
# Signals & lifecycle
# ------------------------
def _signal_handler(signum, _frame):
    if not STOP.is_set():
        log(f"\n[SIGNAL] {signal.Signals(signum).name} received; stopping…")
        STOP.set()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def start_all():
    # start CSI (if available)
    if PICAM_AVAILABLE:
        csis.start_all()
    else:
        log(f"[CSI] disabled: { _IMPORT_ERR if '_IMPORT_ERR' in globals() else 'unavailable' }")
    # start RTSP cams
    if OPENCV_AVAILABLE:
        rtsps.start_all(RTSP_SOURCES)
    else:
        log(f"[RTSP] disabled: { _CV_ERR if '_CV_ERR' in globals() else 'opencv unavailable' }")

def stop_all():
    # stop RTSP first (they can be pulling network frames)
    try: rtsps.stop_all()
    except Exception: pass
    # then CSI
    try: csis.stop_all()
    except Exception: pass

def cleanup_and_exit():
    log("[CLEANUP] stopping…")
    STOP.set()
    stop_all()
    # join workers (streams run daemon threads; best-effort)
    for t in WORKERS:
        try: t.join(timeout=1.0)
        except Exception: pass
    log("[CLEANUP] done.")

def main():
    log("[BOOT] synctankone starting…")
    start_all()

    server = FlaskServerThread(app, HOST, PORT)
    server.start()
    log(f"[READY] http://{HOST}:{PORT}")

    try:
        while not STOP.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        STOP.set()
        log("\n[INTERRUPT] Ctrl-C")
    finally:
        try:
            server.shutdown()
            server.join(timeout=2.0)
        except Exception:
            pass
        cleanup_and_exit()

if __name__ == "__main__":
    main()
