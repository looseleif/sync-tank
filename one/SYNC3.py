#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ROV Dashboard (decoupled control loop + axis toggles + light RTSP snapshots):
- Primary: USB cam (always on) + motor controls (PCA9685 over I2C)
- Secondary: RTSP cams (now snapshot once every RTSP_SNAPSHOT_SEC, default 5s)
- Health endpoints + optional hunter to discover a 3rd RTSP
- MotorController thread runs at fixed rate so RTSP work can't starve PWM.
- Autonomous "circle" explorer mode with per-axis toggles (/auto/axes).
"""

import os, sys, time, signal, subprocess, glob, math
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from flask import Flask, jsonify, Response, render_template_string, request
import cv2

print("[BOOT] reef_dash starting...", flush=True)

# ------------------ GLOBAL CONFIG ------------------
# Network / page
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

# Primary USB camera capture settings
FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

# Carousel timing for IP cams (JS side refresh of <img>)
CAM_SWAP_SEC = float(os.environ.get("CAM_SWAP_SEC", "5.0"))  # default 5s to align with snapshots

# RTSP snapshot cadence (worker only grabs a frame every N seconds)
RTSP_SNAPSHOT_SEC = float(os.environ.get("RTSP_SNAPSHOT_SEC", "5.0"))

# Known RTSP URLs (edit these to your working trio)
KNOWN_URLS = [
    "rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/101",
    "rtsp://admin:admin@192.168.0.108:554/h264/ch1/main/av_stream",
]

# Optional hunter target: e.g. export HUNT_TARGET=192.168.0.108:554
HUNT_TARGET = os.environ.get("HUNT_TARGET", "").strip()

# I2C / PWM
USE_PWM = True
try:
    import smbus2
except Exception:
    USE_PWM = False

# Force OpenCV/FFmpeg to use TCP for RTSP unless overridden + short timeouts
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|rw_timeout;1500000|stimeout;1500000")

# Motors: 3 channels with midpoint and step
MOTORS = [
    {"ch": 0, "mid": 400, "nudge": 30, "pos": 400},  # forward/back (X)
    {"ch": 4, "mid": 400, "nudge": 30, "pos": 400},  # left/right (Y)
    {"ch": 8, "mid": 500, "nudge": 30, "pos": 400},  # up/down (Z)
]
CLAMP = (0, 4095)
CMD_MAP: Dict[str, Optional[tuple]] = {
    "w": (0, +1), "s": (0, -1),
    "a": (1, -1), "d": (1, +1),
    "q": (2, +1), "e": (2, -1),
    "stop": None,
}

# ------------------ APP STATE ------------------
app = Flask(__name__)
_stop = threading.Event()

# ------------------ UTIL: FrameBuffer ------------------
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

# ------------------ USB CAMERA ------------------
_usb_cap_lock = threading.Lock()
_usb_label = "<none>"
_usb_buf = FrameBuffer()

def discover_by_id_nodes() -> List[str]:
    nodes = sorted(
        p for p in glob.glob("/dev/v4l/by-id/*-video-index0")
        if os.path.islink(p) or os.path.exists(p)
    )
    if nodes:
        return [os.path.realpath(p) for p in nodes]
    # fallback
    return [p for p in ["/dev/video0","/dev/video1","/dev/video2","/dev/video3"] if os.path.exists(p)]

def open_uvc(node: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # offload if supported
    if not cap.isOpened():
        try: cap.release()
        except: pass
        return None
    return cap

def usb_worker():
    global _usb_label
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    nodes = discover_by_id_nodes()
    if not nodes:
        print("[USB] No video devices found", flush=True)
        return
    # Use first that opens
    cap = None
    for n in nodes:
        c = open_uvc(n)
        if c:
            cap = c
            _usb_label = os.path.basename(n)
            print(f"[USB] opened {n}", flush=True)
            break
    if cap is None:
        print("[USB] No cameras could be opened", flush=True)
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    while not _stop.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue
        # size check
        if FRAME_W and FRAME_H:
            h, w = frame.shape[:2]
            if abs(w-FRAME_W) > 8 or abs(h-FRAME_H) > 8:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        # overlay label
        try:
            cv2.putText(frame, f"USB: {_usb_label}", (12, 28), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
        except Exception:
            pass
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            _usb_buf.set(jpg.tobytes())

    try: cap.release()
    except: pass
    print("[USB] worker stopped", flush=True)

# ------------------ I2C / PWM (PCA9685) ------------------
class PWM:
    def __init__(self, addr=0x40, bus_id=1):
        self.ok = False
        self.addr = addr
        self.bus = None
        if not USE_PWM:
            print("[PWM] smbus2 not available; PWM disabled", flush=True)
            return
        try:
            self.bus = smbus2.SMBus(bus_id)
            MODE1, PRESCALE = 0x00, 0xFE
            # Wake
            self.bus.write_byte_data(addr, MODE1, 0x00)
            time.sleep(0.005)
            # 50 Hz
            prescale_val = int(25000000.0 / (4096 * 50) - 1)
            self.bus.write_byte_data(addr, MODE1, 0x10)
            self.bus.write_byte_data(addr, PRESCALE, prescale_val)
            self.bus.write_byte_data(addr, MODE1, 0x00)
            time.sleep(0.005)
            # Auto-increment + all-call
            self.bus.write_byte_data(addr, MODE1, 0xA1)
            self.ok = True
            print("[PWM] initialized OK", flush=True)
        except Exception as e:
            print(f"[PWM] Disabled (I2C error): {e}", flush=True)
            self.ok = False

    def set_pwm(self, ch: int, on: int, off: int):
        if not self.ok:
            return
        off = max(CLAMP[0], min(CLAMP[1], off))
        base = 0x06 + 4 * ch
        try:
            self.bus.write_byte_data(self.addr, base + 0, on & 0xFF)
            self.bus.write_byte_data(self.addr, base + 1, (on >> 8) & 0xFF)
            self.bus.write_byte_data(self.addr, base + 2, off & 0xFF)
            self.bus.write_byte_data(self.addr, base + 3, (off >> 8) & 0xFF)
        except Exception as e:
            # Mark unhealthy on write failure so health shows it
            self.ok = False
            print(f"[PWM] write error: {e}", flush=True)

# global PWM and write lock
pwm_lock = threading.Lock()
pwm = PWM()
for m in MOTORS:
    m["pos"] = m["mid"]
    with pwm_lock:
        pwm.set_pwm(m["ch"], 0, m["mid"])

def pwm_watchdog():
    """
    Very light-touch health ping: reads MODE1 register to ensure I2C alive,
    and re-inits if we detect a failure.
    """
    MODE1 = 0x00
    while not _stop.is_set():
        time.sleep(2.0)
        if not USE_PWM:
            continue
        try:
            if pwm.bus is None:
                raise IOError("bus is None")
            _ = pwm.bus.read_byte_data(pwm.addr, MODE1)
            # If read works, ensure ok flag is True
            pwm.ok = True
        except Exception as e:
            print(f"[PWM] watchdog read failed, attempting re-init: {e}", flush=True)
            try:
                addr = pwm.addr
                _new = PWM(addr=addr)
                globals()["pwm"] = _new
            except Exception as ee:
                print(f"[PWM] re-init failed: {ee}", flush=True)

# ------------------ MOTOR CONTROLLER (decoupled loop) ------------------
@dataclass
class AutoConfig:
    radius_xy: int = 120        # +/- ticks around mid for XY circle
    radius_z: int = 40          # gentle bobbing
    period_s: float = 12.0      # seconds per XY circle
    z_period_s: float = 7.0     # seconds per up/down cycle
    slew_per_tick: int = 12     # how fast to chase target (ticks per control step)
    tick_hz: float = 20.0       # control loop frequency
    enable_x: bool = True       # explore forward/back
    enable_y: bool = True       # explore left/right
    enable_z: bool = False      # explore up/down (off by default)

class MotorController(threading.Thread):
    def __init__(self, motors, pwm, lock):
        super().__init__(daemon=True)
        self.motors = motors
        self.pwm = pwm
        self.lock = lock
        self._stop = threading.Event()
        self.mode = "manual"   # or "auto"
        self.auto_cfg = AutoConfig()
        self._t0 = time.time()
        # desired target positions (the loop slews toward these)
        self.targets = [m["mid"] for m in motors]

    def stop(self): self._stop.set()

    def set_mode(self, mode: str):
        self.mode = "auto" if mode.lower().startswith("a") else "manual"
        # snap targets to current pos on mode switch
        self.targets = [int(m["pos"]) for m in self.motors]

    def set_target_nudge(self, axis: int, sign: int):
        m = self.motors[axis]
        lower = m["mid"] - m["nudge"] * 5
        upper = m["mid"] + m["nudge"] * 5
        tgt = self.targets[axis] + (m["nudge"] * sign)
        self.targets[axis] = max(lower, min(upper, tgt))

    def set_all_mid(self):
        for i, m in enumerate(self.motors):
            self.targets[i] = m["mid"]

    def set_auto_params(self, **kw):
        for k, v in kw.items():
            if hasattr(self.auto_cfg, k):
                # cast by semantic
                if k in ("period_s", "z_period_s", "tick_hz"):
                    setattr(self.auto_cfg, k, float(v))
                elif k.startswith("enable_"):
                    setattr(self.auto_cfg, k, bool(v))
                else:
                    setattr(self.auto_cfg, k, int(v))

    def _compute_auto_targets(self, now: float):
        cfg = self.auto_cfg
        # Center points
        mx0, my0, mz0 = self.motors[0]["mid"], self.motors[1]["mid"], self.motors[2]["mid"]
        # Phase angles
        theta = 2*math.pi*((now - self._t0) / max(0.1, cfg.period_s))
        zeta  = 2*math.pi*((now - self._t0) / max(0.1, cfg.z_period_s))

        # XY circle (0: fwd/back, 1: left/right)
        tx = mx0 + int(cfg.radius_xy * math.cos(theta)) if cfg.enable_x else mx0
        ty = my0 + int(cfg.radius_xy * math.sin(theta)) if cfg.enable_y else my0
        tz = mz0 + int(cfg.radius_z  * math.sin(zeta))  if cfg.enable_z else mz0

        self.targets = [tx, ty, tz]

    def run(self):
        dt = 1.0 / self.auto_cfg.tick_hz
        while not self._stop.is_set():
            start = time.time()
            # Compute targets for auto
            if self.mode == "auto":
                self._compute_auto_targets(start)

            # Slew positions toward targets and write PWM once per axis
            for i, m in enumerate(self.motors):
                cur = m["pos"]
                tgt = int(self.targets[i])
                if cur == tgt:
                    out = cur
                else:
                    step = self.auto_cfg.slew_per_tick
                    if tgt > cur: out = min(cur + step, tgt)
                    else:         out = max(cur - step, tgt)
                m["pos"] = out
                with self.lock:
                    self.pwm.set_pwm(m["ch"], 0, int(out))

            # Sleep to maintain fixed rate
            elapsed = time.time() - start
            time.sleep(max(0.0, dt - elapsed))

# ------------------ RTSP CAMS (snapshot every N seconds) ------------------
class CamWorker(threading.Thread):
    """
    Light-touch RTSP worker:
    - Opens RTSP, grabs a single frame, encodes JPEG, updates buffer.
    - Releases capture and sleeps RTSP_SNAPSHOT_SEC.
    This avoids continuous decode and keeps CPU free for control loop.
    """
    def __init__(self, idx: int, url: str, title: Optional[str] = None):
        super().__init__(daemon=True)
        self.idx = idx
        self.url = url
        self.title = title or url
        self.buf = FrameBuffer()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # Try to minimize internal buffering if backend honors it
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if cap.isOpened():
            return cap
        return None

    def _grab_one_frame(self) -> Optional[bytes]:
        cap = self._open_capture()
        if not cap:
            return None
        try:
            if FRAME_W: cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            if FRAME_H: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            if FRAME_FPS: cap.set(cv2.CAP_PROP_FPS, min(FRAME_FPS, 5))  # low fps hint

            # Grab a few frames to get a fresh keyframe if needed
            ok, frame = cap.read()
            if not ok or frame is None:
                return None

            if FRAME_W and FRAME_H:
                h, w = frame.shape[:2]
                if abs(w - FRAME_W) > 8 or abs(h - FRAME_H) > 8:
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

            ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok2:
                return jpg.tobytes()
            return None
        finally:
            try: cap.release()
            except: pass

    def run(self):
        print(f"[CAM{self.idx}] snapshot worker for {self.url}", flush=True)
        while not self._stop.is_set():
            start = time.time()
            jpg = self._grab_one_frame()
            if jpg:
                self.buf.set(jpg)
            else:
                print(f"[CAM{self.idx}] snapshot failed; will retry", flush=True)
            # Sleep the remainder up to RTSP_SNAPSHOT_SEC (never block control loop)
            elapsed = time.time() - start
            to_sleep = max(0.1, RTSP_SNAPSHOT_SEC - elapsed)
            for _ in range(int(to_sleep * 10)):
                if self._stop.is_set():
                    break
                time.sleep(0.1)
        print(f"[CAM{self.idx}] stopped", flush=True)

class CamRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cams: Dict[int, CamWorker] = {}
        self._next_idx = 0

    def add(self, url: str, title: Optional[str] = None) -> int:
        with self._lock:
            if any(c.url == url for c in self._cams.values()):
                for i, c in self._cams.items():
                    if c.url == url:
                        return i
            idx = self._next_idx
            self._next_idx += 1
            worker = CamWorker(idx, url, title or f"RTSP {idx}")
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

    def stop_all(self):
        with self._lock():
            for c in self._cams.values():
                c.stop()

cams = CamRegistry()

# Hunter (optional)
HUNT_PATHS = [
    "/h264/ch1/main/av_stream",
    "/live/ch00_0", "/live", "/videoMain",
    "/Streaming/Channels/101",
    "/h264Preview_01_main",
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
]
HUNT_CREDS = [("admin","admin"), ("admin","123456"), ("admin","")]

def ffprobe_video(url: str, transport: str = "tcp", timeout_s: float = 3.0) -> bool:
    args = [
        "ffprobe","-v","error","-hide_banner",
        "-rtsp_transport", transport,
        "-select_streams","v:0",
        "-show_entries","stream=codec_type",
        "-of","default=nokey=1:noprint_wrappers=1",
        url
    ]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        return (out.returncode == 0 and "video" in out.stdout)
    except Exception:
        return False

def hunter_thread(hostport: str):
    print(f"[HUNT] starting for {hostport}", flush=True)
    host, _, port = hostport.partition(":")
    port = port or "554"
    tried = 0
    candidates: List[str] = []
    for u,pw in HUNT_CREDS:
        for path in HUNT_PATHS:
            candidates.append(f"rtsp://{u}:{pw}@{host}:{port}{path}")
    for path in HUNT_PATHS:
        candidates.append(f"rtsp://{host}:{port}{path}")  # anon

    for url in candidates:
        tried += 1
        if ffprobe_video(url, "udp", 2.5) or ffprobe_video(url, "tcp", 2.5):
            cams.add(url, f"Hunted @ {hostport}")
            print(f"[HUNT] FOUND {url}", flush=True)
            return
        if tried % 10 == 0:
            print(f"[HUNT] tried {tried} candidates...", flush=True)
    print("[HUNT] done; none found", flush=True)

# Carousel state for IP cams (just for UI; switching is done client-side)
_carousel_idx = 0
_carousel_lock = threading.Lock()

def carousel_ticker():
    global _carousel_idx
    while not _stop.is_set():
        time.sleep(CAM_SWAP_SEC)
        lst = cams.list()
        if len(lst) > 1:
            with _carousel_lock:
                _carousel_idx = (_carousel_idx + 1) % len(lst)

# ------------------ FLASK PAGES ------------------
PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>REEFLEX — Unified ROV Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; background:#0b0b0b; color:#eee; font-family: system-ui, sans-serif; }
    header { padding:.75rem 1rem; background:#161616; border-bottom:1px solid #222; position:sticky; top:0 }
    main { padding:12px; display:grid; gap:12px; max-width:1200px; margin:0 auto; }
    .hero { display:grid; gap:10px; }
    .controls { display:flex; flex-wrap:wrap; gap:8px }
    button { padding:.55rem .85rem; background:#262626; color:#eee; border:1px solid #333; border-radius:8px; cursor:pointer; }
    button:hover { background:#2e2e2e }
    img.main { width:100%; max-height:65vh; object-fit:contain; background:#000; border:1px solid #222; border-radius:10px }
    .row { display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start; }
    .card { flex:1 1 320px; background:#000; border:1px solid #1f1f1f; border-radius:10px; overflow:hidden; }
    .card h2 { margin:0; padding:8px 12px; background:#222; font-size:14px; border-bottom:1px solid #1f1f1f; }
    .card .sub { color:#9aa; font-size:12px; padding:0 12px 8px; overflow-wrap:anywhere; }
    .card img { width:100%; aspect-ratio:16/9; object-fit:contain; display:block; background:#000; }
    small { color:#9aa }
    .pill { display:inline-block; padding:.15rem .5rem; border:1px solid #333; border-radius:999px; margin-left:6px; font-size:12px; }
    .ok { color:#9f9; border-color:#2a4; }
    .bad{ color:#f99; border-color:#933; }
    pre { background:#111; border:1px solid #222; padding:8px 10px; border-radius:8px; white-space:pre-wrap; }
    a { color:#7fc7ff; }
    label { font-size:12px; color:#bcc; margin-right:6px; }
    .toggles { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  </style>
</head>
<body>
  <header>
    <strong>REEFLEX — Unified ROV Dashboard</strong>
    <span class="pill" id="pwm">PWM: ?</span>
    <span class="pill" id="cams">RTSP: ?</span>
    <span class="pill">USB: {{usb_label}}</span>
    <span style="float:right; color:#9aa">http://{{host}}:{{port}}</span>
  </header>

  <main>
    <section class="hero">
      <img class="main" src="/usb" />
      <div class="controls">
        <button onclick="send('w')">Forward (W)</button>
        <button onclick="send('s')">Back (S)</button>
        <button onclick="send('a')">Left (A)</button>
        <button onclick="send('d')">Right (D)</button>
        <button onclick="send('q')">Up (Q)</button>
        <button onclick="send('e')">Down (E)</button>
        <button onclick="send('stop')">Stop</button>
        <button onclick="fetch('/auto/start').then(()=>status())">Auto: Start</button>
        <button onclick="fetch('/auto/stop').then(()=>status())">Auto: Stop</button>
      </div>
      <div class="toggles">
        <label><input type="checkbox" id="ax_x" checked onchange="pushAxes()"> Explore X (F/B)</label>
        <label><input type="checkbox" id="ax_y" checked onchange="pushAxes()"> Explore Y (L/R)</label>
        <label><input type="checkbox" id="ax_z" onchange="pushAxes()"> Explore Z (Up/Down)</label>
      </div>
      <small>
        Keys: W/A/S/D/Q/E, Space or X = Stop.
        Health: <a href="/health" target="_blank">/health</a>.
        RTSP snapshots every {{swap}}s.
      </small>
      <pre id="status"></pre>
    </section>

    <section>
      <h3>IP Cameras (refresh every {{swap}}s)</h3>
      <div class="row" id="iprow">
        <!-- JS populates cards -->
      </div>
    </section>
  </main>

<script>
async function send(cmd){
  try {
    const r = await fetch('/cmd/'+cmd);
    const t = await r.text();
    document.getElementById('status').textContent = t;
  } catch(e) {}
}
document.addEventListener('keydown', (e)=>{
  const k = e.key.toLowerCase();
  if(['w','a','s','d','q','e'].includes(k)) send(k);
  if(k==='x' || k===' ') send('stop');
});

async function status(){
  try{
    const r = await fetch('/health');
    const j = await r.json();
    const pwm = document.getElementById('pwm');
    pwm.textContent = 'PWM: ' + (j.pwm_ok ? 'OK' : 'ERR') + ' ('+j.mode+')';
    pwm.className = 'pill ' + (j.pwm_ok ? 'ok':'bad');

    const cams = document.getElementById('cams');
    cams.textContent = 'RTSP: ' + j.rtsp_count;
    cams.className = 'pill ' + (j.rtsp_count>0 ? 'ok':'bad');

    // reflect axis toggles if health reports them
    if (j.auto_axes) {
      document.getElementById('ax_x').checked = !!j.auto_axes.enable_x;
      document.getElementById('ax_y').checked = !!j.auto_axes.enable_y;
      document.getElementById('ax_z').checked = !!j.auto_axes.enable_z;
    }

    document.getElementById('status').textContent = JSON.stringify(j, null, 2);
  }catch(e){}
  setTimeout(status, 2000);
}
status();

async function pushAxes(){
  const x = document.getElementById('ax_x').checked;
  const y = document.getElementById('ax_y').checked;
  const z = document.getElementById('ax_z').checked;
  try{
    await fetch('/auto/axes', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({enable_x:x, enable_y:y, enable_z:z})
    });
  }catch(e){}
}

let cams = [];
async function loadCams(){
  try{
    const r = await fetch('/cams');
    const j = await r.json();
    cams = j.items || [];
    renderCamCards();
  }catch(e){}
}
function renderCamCards(){
  const row = document.getElementById('iprow');
  row.innerHTML = '';
  cams.forEach((c, idx)=>{
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h2>${c.title}</h2>
      <div class="sub">${c.url}</div>
      <img id="rtsp_${c.idx}" src="/cam/${c.idx}" />
    `;
    row.appendChild(card);
  });
}
loadCams();

// Every swap seconds, reload one <img> to pick up the latest snapshot (cheap).
let cur = 0;
setInterval(()=>{
  if(cams.length===0) return;
  const c = cams[cur % cams.length];
  const img = document.getElementById('rtsp_'+c.idx);
  if(img){
    img.src = '/cam/'+c.idx+'?t='+(Date.now());
  }
  cur++;
}, {{swap_ms}});

</script>
</body>
</html>
"""

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template_string(
        PAGE,
        host=HOST, port=PORT,
        usb_label=_usb_label,
        swap=CAM_SWAP_SEC,
        swap_ms=int(CAM_SWAP_SEC*1000)
    )

@app.route("/usb")
def usb_stream():
    def gen():
        boundary = b"--frame\r\n"
        while not _stop.is_set():
            frame = _usb_buf.get()
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cams")
def list_cams():
    items = [{"idx": i, "url": u, "title": t} for i, u, t in cams.list()]
    return jsonify({"items": items, "count": len(items)})

@app.route("/cam/<int:idx>")
def stream_cam(idx: int):
    worker = cams.get(idx)
    if worker is None:
        return "Camera not found", 404
    def gen():
        boundary = b"--frame\r\n"
        # Serve last snapshot as an MJPEG that updates when worker writes a new one.
        while not _stop.is_set():
            frame = worker.buf.get()
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            # small delay so we don't spin; actual update cadence is in worker/JS reload
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cmd/<c>")
def cmd(c: str):
    c = c.lower()
    if c not in CMD_MAP and c != "stop":
        return f"ERR unknown cmd {c}", 400
    if c == "stop":
        control.set_all_mid()
        return "OK stop -> midpoints (targets)"
    idx_sign = CMD_MAP[c]
    if idx_sign is None:
        control.set_all_mid()
        return "OK stop -> midpoints (targets)"
    idx, sign = idx_sign
    control.set_mode("manual")  # touching keys implies manual intent
    control.set_target_nudge(idx, sign)
    m = MOTORS[idx]
    return f"OK {c}: ch={m['ch']} target={int(control.targets[idx])} pos={int(m['pos'])}"

@app.route("/mode/<m>")
def set_mode(m: str):
    control.set_mode(m)
    return jsonify({"ok": True, "mode": control.mode})

@app.route("/auto/start")
def auto_start():
    control.set_mode("auto")
    return jsonify({"ok": True, "mode": control.mode})

@app.route("/auto/stop")
def auto_stop():
    control.set_mode("manual")
    control.set_all_mid()
    return jsonify({"ok": True, "mode": control.mode})

@app.route("/auto/conf", methods=["POST"])
def auto_conf():
    j = request.get_json(force=True, silent=True) or {}
    kw = {}
    for k in ["radius_xy", "radius_z", "period_s", "z_period_s", "slew_per_tick", "tick_hz",
              "enable_x", "enable_y", "enable_z"]:
        if k in j:
            kw[k] = j[k]
    control.set_auto_params(**kw)
    return jsonify({"ok": True, "cfg": control.auto_cfg.__dict__})

@app.route("/auto/axes", methods=["POST"])
def auto_axes():
    j = request.get_json(force=True, silent=True) or {}
    kw = {}
    for k in ["enable_x", "enable_y", "enable_z"]:
        if k in j:
            kw[k] = bool(j[k])
    control.set_auto_params(**kw)
    return jsonify({"ok": True, "cfg": control.auto_cfg.__dict__})

@app.route("/health")
def health():
    auto_axes = {
        "enable_x": control.auto_cfg.enable_x,
        "enable_y": control.auto_cfg.enable_y,
        "enable_z": control.auto_cfg.enable_z,
    } if 'control' in globals() else {}
    return jsonify({
        "pwm_ok": bool(pwm.ok),
        "mode": getattr(globals().get("control", None), "mode", "unknown"),
        "motors": [{"ch": m["ch"], "pos": int(m["pos"]), "mid": m["mid"]} for m in MOTORS],
        "targets": getattr(globals().get("control", None), "targets", []),
        "auto_axes": auto_axes,
        "usb_label": _usb_label,
        "frame_w": FRAME_W, "frame_h": FRAME_H, "fps": FRAME_FPS,
        "swap_sec": CAM_SWAP_SEC,
        "rtsp_snapshot_sec": RTSP_SNAPSHOT_SEC,
        "rtsp_count": len(cams.list())
    })

@app.route("/add", methods=["POST"])
def add_cam():
    j = request.get_json(force=True, silent=True) or {}
    url = j.get("url")
    title = j.get("title")
    if not url:
        return jsonify({"ok": False, "error": "missing url"}), 400
    idx = cams.add(url, title)
    return jsonify({"ok": True, "idx": idx})

# ------------------ LIFECYCLE ------------------
def start_threads():
    # USB worker
    threading.Thread(target=usb_worker, daemon=True).start()
    # PWM watchdog
    threading.Thread(target=pwm_watchdog, daemon=True).start()
    # RTSP carousel ticker (just updates which image to refresh on the UI)
    # threading.Thread(target=carousel_ticker, daemon=True).start()
    # Motor controller (decoupled from Flask & cameras)
    globals()["control"] = MotorController(MOTORS, pwm, pwm_lock)
    control.start()

def start_rtsp():
    # known cams
    for url in KNOWN_URLS:
        cams.add(url, f"Fixed: {url}")
    # optional hunter
    if HUNT_TARGET:
        threading.Thread(target=hunter_thread, args=(HUNT_TARGET,), daemon=True).start()
        print(f"[MAIN] hunter launched for {HUNT_TARGET}", flush=True)
    else:
        print("[MAIN] no hunter target configured (set HUNT_TARGET=ip:port to enable).", flush=True)

def cleanup_and_exit(*_):
    _stop.set()
    try:
        control.stop()
    except Exception:
        pass
    try:
        for m in MOTORS:
            with pwm_lock:
                pwm.set_pwm(m["ch"], 0, m["mid"])
    except:
        pass
    try:
        cams.stop_all()
    except Exception:
        pass
    print("\n[EXIT] cleaned up.", flush=True)
    # Give threads a moment to unwind
    try: time.sleep(0.3)
    except: pass
    os._exit(0)

def main():
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    # start_rtsp()
    start_threads()

    print(f"[READY] http://{HOST}:{PORT}  (USB at /usb, RTSP at /cam/<idx>, add via POST /add)", flush=True)
    app.run(host=HOST, port=PORT, threaded=True)

if __name__ == "__main__":
    main()
