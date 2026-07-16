#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ROV Dashboard — USB (robust) + CSI (Picamera2) + PWM
- USB: robust discovery (USB-only), primary selection, warmup, MJPG hint.
- CSI: Picamera2 continuous capture (headless), Pillow→JPEG, MJPEG endpoints.
- Dashboard shows USB + CSI tiles; hero = USB PRIMARY.
- Motor controller decoupled; camera threads never block PWM.

ENV knobs:
  HOST=0.0.0.0 PORT=8080
  FRAME_W=1280 FRAME_H=720 FRAME_FPS=15 JPEG_QUALITY=80     # USB defaults
  MAX_USB_CAMS=3                                            # main + up to 2 secondaries
  USB_PRIMARY=0                                             # which discovered USB index is the hero
  CSI_INDEXES=0,1                                           # which CSI indexes to run (defaults to "all detected" if empty)
  CSI_W=1280 CSI_H=720 CSI_FPS=15 CSI_JPEG_QUALITY=70
  USE_PWM=1                                                 # set 0 to hard-disable PWM init
"""

import os, sys, time, signal, subprocess, glob, math, shutil, io
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from flask import Flask, jsonify, Response, render_template_string, request, abort
import cv2
import numpy as np

# --- CSI deps (Picamera2 + Pillow) ---
try:
    from picamera2 import Picamera2
    from PIL import Image
    PICAM_AVAILABLE = True
except Exception as _e:
    PICAM_AVAILABLE = False

print("[BOOT] reef_dash starting...", flush=True)

# ------------------ GLOBAL CONFIG ------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

MAX_USB_CAMS = int(os.environ.get("MAX_USB_CAMS", "3"))
USB_PRIMARY = int(os.environ.get("USB_PRIMARY", "0"))

CSI_INDEXES_ENV = [s.strip() for s in os.environ.get("CSI_INDEXES", "0,1").split(",") if s.strip() != ""]
CSI_W = int(os.environ.get("CSI_W", "1280"))
CSI_H = int(os.environ.get("CSI_H", "720"))
CSI_FPS = int(os.environ.get("CSI_FPS", "15"))
CSI_JPEG_QUALITY = int(os.environ.get("CSI_JPEG_QUALITY", "70"))

USE_PWM_ENV = os.environ.get("USE_PWM", "1").lower() not in ("0", "false", "no")

# ------------------ APP ------------------
app = Flask(__name__)
_stop = threading.Event()

# ------------------ UTIL ------------------
class FrameBuffer:
    def __init__(self, prime_black: bool = False, w: int = 320, h: int = 180, q: int = 60, label: str = ""):
        self._buf: Optional[bytes] = None
        self._cond = threading.Condition()
        if prime_black:
            self._buf = self._make_black(w, h, q, label)

    def _make_black(self, w, h, q, label) -> bytes:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        if label:
            cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return jpg.tobytes() if ok else b""

    def set(self, b: bytes):
        with self._cond:
            self._buf = b
            self._cond.notify_all()

    def get(self) -> bytes:
        with self._cond:
            while self._buf is None:
                self._cond.wait()
            return self._buf

# ------------------ USB CAMS (robust discovery + primary selection) ------------------
def _by_path_symlinks() -> List[str]:
    paths = sorted(glob.glob("/dev/v4l/by-path/*-usb-*-video-index0"))
    return [os.path.realpath(p) for p in paths if os.path.exists(p)]

def _fallback_video_nodes() -> List[str]:
    nodes = []
    for n in range(0, 64):
        dev = f"/dev/video{n}"
        if os.path.exists(dev):
            nodes.append(dev)
    return nodes

def _is_usb_node(node: str) -> bool:
    for p in glob.glob("/dev/v4l/by-path/*-usb-*-video-index0"):
        try:
            if os.path.realpath(p) == node:
                return True
        except Exception:
            pass
    return False

def discover_usb_nodes(max_n: int) -> List[str]:
    nodes = _by_path_symlinks()
    if not nodes:
        fb = _fallback_video_nodes()
        nodes = [n for n in fb if _is_usb_node(n)]
    uniq = []
    seen = set()
    for n in nodes:
        if n not in seen:
            uniq.append(n); seen.add(n)
        if len(uniq) >= max_n:
            break
    return uniq

class USBWorker(threading.Thread):
    def __init__(self, idx: int, node: str):
        super().__init__(daemon=True)
        self.idx = idx
        self.node = node
        self.buf = FrameBuffer(prime_black=True, w=FRAME_W, h=FRAME_H, label=f"USB{idx}: {os.path.basename(node)}")
        self._stop = threading.Event()
        self.last_error = ""
        self.ok = False

    def stop(self): self._stop.set()

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.node, cv2.CAP_V4L2)
        if not cap.isOpened():
            try: cap.release()
            except: pass
            self.last_error = "open failed"
            return None
        try:
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
            try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception: pass
        except Exception as e:
            self.last_error = f"cfg error: {e}"
        return cap

    def _warmup(self, cap, n=5):
        for _ in range(n):
            ok, _frm = cap.read()
            if self._stop.is_set(): break
            if not ok: time.sleep(0.02)

    def run(self):
        print(f"[USB{self.idx}] opening {self.node}", flush=True)
        tries = 0
        cap = None
        while not self._stop.is_set() and tries < 3:
            cap = self._open()
            if cap: break
            tries += 1
            time.sleep(0.25)
        if not cap:
            print(f"[USB{self.idx}] open failed permanently: {self.last_error}", flush=True)
            return
        self._warmup(cap, n=5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        self.ok = True
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                self.last_error = "read failed"
                time.sleep(0.02)
                continue
            h, w = frame.shape[:2]
            if FRAME_W and FRAME_H and (abs(w-FRAME_W) > 8 or abs(h-FRAME_H) > 8):
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            try:
                cv2.putText(frame, f"USB{self.idx}: {os.path.basename(self.node)}", (12, 28), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
            except Exception:
                pass
            ok2, jpg = cv2.imencode(".jpg", frame, enc)
            if ok2:
                self.buf.set(jpg.tobytes())
        try: cap.release()
        except: pass
        print(f"[USB{self.idx}] stopped", flush=True)

class USBRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cams: Dict[int, USBWorker] = {}
        self._nodes: List[str] = []
        self._primary_idx = USB_PRIMARY

    def start(self, max_n: int = 3):
        with self._lock:
            self._nodes = discover_usb_nodes(max_n)
            if not self._nodes:
                print("[USB] No USB cameras discovered (USB-only discovery).", flush=True)
            for idx, node in enumerate(self._nodes):
                w = USBWorker(idx, node)
                self._cams[idx] = w
                w.start()
                print(f"[REG] added USB cam {idx}: {node}", flush=True)
            if self._primary_idx >= len(self._nodes):
                self._primary_idx = 0

    def list(self) -> List[Tuple[int, str]]:
        with self._lock:
            return [(i, self._nodes[i]) for i in sorted(self._cams.keys())]

    def get(self, idx: int) -> Optional[USBWorker]:
        with self._lock:
            return self._cams.get(idx)

    def set_primary(self, idx: int) -> bool:
        with self._lock:
            if idx in self._cams:
                self._primary_idx = idx
                print(f"[USB] primary set to index {idx}", flush=True)
                return True
            return False

    def primary_idx(self) -> int:
        with self._lock:
            return self._primary_idx

    def stop_all(self):
        with self._lock:
            for c in self._cams.values():
                c.stop()

usbs = USBRegistry()

# ------------------ CSI CAMS (Picamera2 continuous → MJPEG) ------------------
class SimpleCSICam:
    """Capture loop that keeps latest JPEG for MJPEG streaming."""
    def __init__(self, cam_index:int, size=(1280,720), fps:int=15, q:int=70):
        self.idx = cam_index
        self.size = size
        self.fps  = max(1, min(int(fps), 30))
        self.q    = max(1, min(int(q), 95))
        self.picam: Optional[Picamera2] = None
        self.latest_jpeg: Optional[bytes] = None
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.ok = False
        self.last_error = ""

    def start(self):
        if not PICAM_AVAILABLE:
            self.last_error = "Picamera2 not available"
            return
        if self._t and self._t.is_alive():
            return
        try:
            self.picam = Picamera2(camera_num=self.idx)
            cfg = self.picam.create_preview_configuration(main={"size": self.size, "format": "RGB888"})
            self.picam.configure(cfg)
            self.picam.start()
            time.sleep(0.3)  # AE/AWB warmup
            self._stop.clear()
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
            self.ok = True
            print(f"[CSI{self.idx}] started ({self.size[0]}x{self.size[1]} @ {self.fps}fps, q={self.q})", flush=True)
        except Exception as e:
            self.last_error = f"start fail: {e}"
            self.ok = False
            try:
                if self.picam: self.picam.close()
            except Exception:
                pass
            self.picam = None

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
        self.ok = False

    def _loop(self):
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                arr = self.picam.capture_array("main")  # numpy array (H,W,3) RGB
                with io.BytesIO() as buff:
                    Image.fromarray(arr).save(buff, format="JPEG", quality=self.q, optimize=True)
                    self.latest_jpeg = buff.getvalue()
            except Exception as e:
                self.last_error = f"cap err: {e}"
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def frames(self):
        boundary = b"--frame\r\n"
        while not _stop.is_set():
            if self.latest_jpeg is None:
                time.sleep(0.05); continue
            jpg = self.latest_jpeg
            yield (boundary +
                   b"Content-Type: image/jpeg\r\n" +
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")

class CSIRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cams: Dict[int, SimpleCSICam] = {}
        self._info: List[str] = []

    def detect(self) -> List[int]:
        """Return list of available indices. Also caches descriptive info strings."""
        if not PICAM_AVAILABLE:
            self._info = []
            return []
        try:
            info = Picamera2.global_camera_info()
            self._info = [str(x) for x in info]  # textual repr of camera info
            return list(range(len(info)))
        except Exception:
            self._info = []
            return []

    def start(self, indexes: Optional[List[int]] = None):
        with self._lock:
            avail = self.detect()
            if indexes is None or len(indexes) == 0:
                use = avail
            else:
                # Only keep those present
                use = [i for i in indexes if i in avail]
            for i in use:
                if i not in self._cams:
                    cam = SimpleCSICam(i, size=(CSI_W, CSI_H), fps=CSI_FPS, q=CSI_JPEG_QUALITY)
                    self._cams[i] = cam
                    cam.start()
                    print(f"[REG] added CSI cam {i}", flush=True)
            if not use:
                print("[CSI] No CSI cameras to start.", flush=True)

    def list(self) -> List[Tuple[int, str]]:
        with self._lock:
            # pair idx with best-effort description
            avail = self.detect()
            out = []
            for i in sorted(self._cams.keys()):
                desc = self._info[i] if (self._info and i < len(self._info)) else ""
                if i in avail:
                    out.append((i, desc))
            return out

    def get(self, idx: int) -> Optional[SimpleCSICam]:
        with self._lock:
            return self._cams.get(idx)

    def stop_all(self):
        with self._lock:
            for c in self._cams.values():
                c.stop()

csis = CSIRegistry()

# ------------------ PWM / MOTOR CONTROL ------------------
USE_PWM = USE_PWM_ENV
try:
    import smbus2
except Exception:
    USE_PWM = False

MOTORS = [
    {"ch": 0, "mid": 400, "nudge": 30, "pos": 400},  # X (forward/back)
    {"ch": 4, "mid": 400, "nudge": 30, "pos": 400},  # Y (left/right)
    {"ch": 8, "mid": 400, "nudge": 30, "pos": 400},  # Z (up/down)
]
CLAMP = (0, 4095)

class PWM:
    def __init__(self, addr=0x40, bus_id=1):
        self.ok = False
        self.addr = addr
        self.bus = None
        if not USE_PWM:
            print("[PWM] disabled by env or smbus2 missing", flush=True)
            return
        try:
            self.bus = smbus2.SMBus(bus_id)
            MODE1, PRESCALE = 0x00, 0xFE
            self.bus.write_byte_data(addr, MODE1, 0x00); time.sleep(0.005)
            prescale_val = int(25000000.0 / (4096 * 50) - 1)
            self.bus.write_byte_data(addr, MODE1, 0x10)
            self.bus.write_byte_data(addr, PRESCALE, prescale_val)
            self.bus.write_byte_data(addr, MODE1, 0x00); time.sleep(0.005)
            self.bus.write_byte_data(addr, MODE1, 0xA1)
            self.ok = True
            print("[PWM] initialized OK", flush=True)
        except Exception as e:
            print(f"[PWM] Disabled (I2C error): {e}", flush=True)
            self.ok = False

    def set_pwm(self, ch: int, on: int, off: int):
        if not self.ok: return
        off = max(CLAMP[0], min(CLAMP[1], off))
        base = 0x06 + 4 * ch
        try:
            self.bus.write_byte_data(self.addr, base + 0, on & 0xFF)
            self.bus.write_byte_data(self.addr, base + 1, (on >> 8) & 0xFF)
            self.bus.write_byte_data(self.addr, base + 2, off & 0xFF)
            self.bus.write_byte_data(self.addr, base + 3, (off >> 8) & 0xFF)
        except Exception as e:
            self.ok = False
            print(f"[PWM] write error: {e}", flush=True)

pwm_lock = threading.Lock()
pwm = PWM()
for m in MOTORS:
    m["pos"] = m["mid"]
    with pwm_lock:
        pwm.set_pwm(m["ch"], 0, m["mid"])

def pwm_watchdog():
    MODE1 = 0x00
    while not _stop.is_set():
        time.sleep(2.0)
        if not USE_PWM: continue
        try:
            if pwm.bus is None: raise IOError("bus is None")
            _ = pwm.bus.read_byte_data(pwm.addr, MODE1)
            pwm.ok = True
        except Exception as e:
            print(f"[PWM] watchdog read failed, attempting re-init: {e}", flush=True)
            try:
                addr = pwm.addr
                globals()["pwm"] = PWM(addr=addr)
            except Exception as ee:
                print(f"[PWM] re-init failed: {ee}", flush=True)

@dataclass
class AutoConfig:
    radius_xy: int = 120
    radius_z: int = 40
    period_s: float = 12.0
    z_period_s: float = 7.0
    slew_per_tick: int = 12
    tick_hz: float = 20.0
    enable_x: bool = True
    enable_y: bool = True
    enable_z: bool = False

class MotorController(threading.Thread):
    def __init__(self, motors, pwm, lock):
        super().__init__(daemon=True)
        self.motors = motors; self.pwm = pwm; self.lock = lock
        self._stop = threading.Event(); self.mode = "manual"
        self.auto_cfg = AutoConfig(); self._t0 = time.time()
        self.targets = [m["mid"] for m in motors]

    def stop(self): self._stop.set()
    def set_mode(self, mode: str):
        self.mode = "auto" if mode.lower().startswith("a") else "manual"
        self.targets = [int(m["pos"]) for m in self.motors]
    def set_target_nudge(self, axis: int, sign: int):
        m = self.motors[axis]
        lower = m["mid"] - m["nudge"] * 5
        upper = m["mid"] + m["nudge"] * 5
        self.targets[axis] = max(lower, min(upper, self.targets[axis] + (m["nudge"] * sign)))
    def set_all_mid(self):
        for i, m in enumerate(self.motors): self.targets[i] = m["mid"]
    def set_auto_params(self, **kw):
        for k, v in kw.items():
            if hasattr(self.auto_cfg, k):
                if k in ("period_s", "z_period_s", "tick_hz"): setattr(self.auto_cfg, k, float(v))
                elif k.startswith("enable_"): setattr(self.auto_cfg, k, (str(v).lower() in ("1","true","yes","on")))
                else: setattr(self.auto_cfg, k, int(v))
    def _compute_auto_targets(self, now: float):
        cfg = self.auto_cfg
        mx0, my0, mz0 = self.motors[0]["mid"], self.motors[1]["mid"], self.motors[2]["mid"]
        theta = 2*math.pi*((now - self._t0) / max(0.1, cfg.period_s))
        zeta  = 2*math.pi*((now - self._t0) / max(0.1, cfg.z_period_s))
        tx = mx0 + int(cfg.radius_xy * math.cos(theta)) if cfg.enable_x else mx0
        ty = my0 + int(cfg.radius_xy * math.sin(theta)) if cfg.enable_y else my0
        tz = mz0 + int(cfg.radius_z  * math.sin(zeta))  if cfg.enable_z else mz0
        self.targets = [tx, ty, tz]
    def run(self):
        dt = 1.0 / self.auto_cfg.tick_hz
        while not self._stop.is_set():
            start = time.time()
            if self.mode == "auto": self._compute_auto_targets(start)
            for i, m in enumerate(self.motors):
                cur = m["pos"]; tgt = int(self.targets[i])
                step = self.auto_cfg.slew_per_tick
                out = cur if cur == tgt else (min(cur + step, tgt) if tgt > cur else max(cur - step, tgt))
                m["pos"] = out
                with self.lock: self.pwm.set_pwm(m["ch"], 0, int(out))
            time.sleep(max(0.0, dt - (time.time() - start)))

# ------------------ HTML ------------------
PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>REEFLEX — ROV Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; background:#0b0b0b; color:#eee; font-family: system-ui, sans-serif; }
    header { padding:.75rem 1rem; background:#161616; border-bottom:1px solid #222; position:sticky; top:0 }
    main { padding:12px; display:grid; gap:16px; max-width:1200px; margin:0 auto; }
    .hero { display:grid; gap:10px; }
    .controls { display:flex; flex-wrap:wrap; gap:8px }
    button { padding:.55rem .85rem; background:#262626; color:#eee; border:1px solid #333; border-radius:8px; cursor:pointer; }
    button:hover { background:#2e2e2e }
    img.main { width:100%; max-height:65vh; object-fit:contain; background:#000; border:1px solid #222; border-radius:10px }
    .row { display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start; }
    .card { flex:1 1 360px; background:#000; border:1px solid #1f1f1f; border-radius:10px; overflow:hidden; }
    .card h2 { margin:0; padding:8px 12px; background:#222; font-size:14px; border-bottom:1px solid #1f1f1f; display:flex; justify-content:space-between; align-items:center; }
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
    .badge { font-size:11px; padding:.1rem .4rem; border:1px solid #444; border-radius:6px; color:#bbb; }
  </style>
</head>
<body>
  <header>
    <strong>REEFLEX — ROV Dashboard</strong>
    <span class="pill" id="pwm">PWM: ?</span>
    <span class="pill" id="usb">USB: ?</span>
    <span class="pill" id="csis">CSI: ?</span>
    <span style="float:right; color:#9aa">http://{{host}}:{{port}}</span>
  </header>

  <main>
    <section class="hero">
      <img class="main" id="hero" src="/usb/stream/PRIMARY" />
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
      <small>Health: <a href="/health" target="_blank">/health</a></small>
      <pre id="status"></pre>
    </section>

    <section>
      <h3>USB Cameras</h3>
      <div class="row" id="usbrow"></div>
    </section>

    <section>
      <h3>CSI Cameras</h3>
      <div class="row" id="csirow"></div>
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

    const usb = document.getElementById('usb');
    usb.textContent = 'USB: ' + j.usb_count + ' (primary ' + j.usb_primary + ')';
    usb.className = 'pill ' + (j.usb_count>0 ? 'ok':'bad');

    const csis = document.getElementById('csis');
    csis.textContent = 'CSI: ' + j.csi_count;
    csis.className = 'pill ' + (j.csi_count>0 ? 'ok':'bad');

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

let usbList = [];
let csiList = [];

async function loadUSBs(){
  try{
    const r = await fetch('/usbs'); const j = await r.json();
    usbList = j.items || [];
    const row = document.getElementById('usbrow'); row.innerHTML = '';
    usbList.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <h2>
          USB ${c.idx} <span class="badge">${c.ok ? 'OK' : '...'}</span>
          <button onclick="setPrimary(${c.idx})">Make Primary</button>
        </h2>
        <div class="sub">${c.node}${c.err ? ' — err: '+c.err : ''}</div>
        <img id="usb_${c.idx}" src="/usb/stream/${c.idx}" />
      `;
      row.appendChild(card);
    });
  }catch(e){}
}

async function setPrimary(idx){
  try{
    await fetch('/usb/primary/'+idx, {method:'POST'});
    document.getElementById('hero').src = '/usb/stream/PRIMARY?t=' + Date.now();
    status();
  }catch(e){}
}

async function loadCSIs(){
  try{
    const r = await fetch('/csis'); const j = await r.json();
    csiList = j.items || [];
    const row = document.getElementById('csirow'); row.innerHTML = '';
    csiList.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <h2>CSI ${c.idx} <span class="badge">${c.ok ? 'OK' : '...'}</span></h2>
        <div class="sub">${c.desc || ''}${c.err ? ' — err: '+c.err : ''}</div>
        <a href="/csi/${c.idx}.mjpg" target="_blank">
          <img id="csi_${c.idx}" src="/csi/${c.idx}.mjpg" />
        </a>
      `;
      row.appendChild(card);
    });
  }catch(e){}
}

loadUSBs();
loadCSIs();

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
</script>
</body>
</html>
"""

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template_string(PAGE, host=HOST, port=PORT)

# ---------- USB: MJPEG streams ----------
@app.route("/usb/stream/<sel>")
def usb_stream(sel: str):
    if sel.upper() == "PRIMARY":
        idx = usbs.primary_idx()
    else:
        try:
            idx = int(sel)
        except:
            return "bad selector", 400
    w = usbs.get(idx)
    if not w:
        return "USB not found", 404
    def gen():
        boundary = b"--frame\r\n"
        while not _stop.is_set():
            frame = w.buf.get()
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/usb/primary/<int:idx>", methods=["POST"])
def set_primary(idx: int):
    ok = usbs.set_primary(idx)
    return jsonify({"ok": ok, "primary": usbs.primary_idx()}), (200 if ok else 404)

@app.route("/usbs")
def list_usbs():
    items = []
    for i, node in usbs.list():
        w = usbs.get(i)
        items.append({"idx": i, "node": node, "ok": bool(w and w.ok), "err": getattr(w, "last_error", "")})
    return jsonify({"items": items, "count": len(items)})

# ---------- CSI: MJPEG streams (Picamera2) ----------
def _csi_stream(idx: int):
    cam = csis.get(idx)
    if not cam:
        abort(404, f"CSI camera {idx} not running")
    return Response(cam.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/csi/<int:idx>.mjpg")
def csi_mjpg(idx: int):
    # Validate against detected set
    detected = [i for i, _ in csis.list()]
    if idx not in detected:
        abort(404, f"CSI camera index {idx} not available")
    return _csi_stream(idx)

# alias without extension for backward-compat with earlier dashboard versions
@app.route("/csi/<int:idx>")
def csi_alias(idx: int):
    detected = [i for i, _ in csis.list()]
    if idx not in detected:
        abort(404, f"CSI camera index {idx} not available")
    return _csi_stream(idx)

@app.route("/csis")
def list_csis():
    items = []
    for i, desc in csis.list():
        w = csis.get(i)
        items.append({
            "idx": i,
            "desc": desc,
            "ok": bool(w and w.ok and w.latest_jpeg is not None),
            "err": getattr(w, "last_error", "")
        })
    return jsonify({"items": items, "count": len(items)})

# ---------- Motor commands & health ----------
CMD_MAP: Dict[str, Optional[tuple]] = {
    "w": (0, +1), "s": (0, -1),
    "a": (1, +1), "d": (1, -1),
    "q": (2, +1), "e": (2, -1),
    "stop": None,
}

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
    control.set_mode("manual")
    control.set_target_nudge(idx, sign)
    m = MOTORS[idx]
    return f"OK {c}: ch={m['ch']} target={int(control.targets[idx])} pos={int(m['pos'])}"

@app.route("/auto/start")
def auto_start():
    control.set_mode("auto")
    return jsonify({"ok": True, "mode": control.mode})

@app.route("/auto/stop")
def auto_stop():
    control.set_mode("manual")
    control.set_all_mid()
    return jsonify({"ok": True, "mode": control.mode})

@app.route("/auto/axes", methods=["POST"])
def auto_axes():
    j = request.get_json(force=True, silent=True) or {}
    kw = {}
    for k in ["enable_x", "enable_y", "enable_z"]:
        if k in j:
            kw[k] = j[k]
    control.set_auto_params(**kw)
    return jsonify({"ok": True, "cfg": control.auto_cfg.__dict__})

@app.route("/health")
def health():
    auto_axes_state = {
        "enable_x": control.auto_cfg.enable_x,
        "enable_y": control.auto_cfg.enable_y,
        "enable_z": control.auto_cfg.enable_z,
    } if 'control' in globals() else {}

    # USB details
    usb_items = []
    for i, node in usbs.list():
        w = usbs.get(i)
        usb_items.append({"idx": i, "node": node, "ok": bool(w and w.ok), "err": getattr(w, "last_error", "")})

    # CSI details
    csi_items = []
    for i, desc in csis.list():
        w = csis.get(i)
        csi_items.append({"idx": i, "desc": desc, "ok": bool(w and w.ok and w.latest_jpeg is not None), "err": getattr(w, "last_error", "")})

    return jsonify({
        "pwm_ok": bool(pwm.ok),
        "mode": getattr(globals().get("control", None), "mode", "unknown"),
        "motors": [{"ch": m["ch"], "pos": int(m["pos"]), "mid": m["mid"]} for m in MOTORS],
        "targets": getattr(globals().get("control", None), "targets", []),
        "auto_axes": auto_axes_state,
        "usb_count": len(usb_items),
        "usb_primary": usbs.primary_idx(),
        "usb": usb_items,
        "csi_count": len(csi_items),
        "csi": csi_items,
        "picamera2": PICAM_AVAILABLE,
        "csi_cfg": {"w": CSI_W, "h": CSI_H, "fps": CSI_FPS, "q": CSI_JPEG_QUALITY}
    })

# ------------------ LIFECYCLE ------------------
def start_threads():
    # USB cams (unchanged)
    usbs.start(MAX_USB_CAMS)

    # CSI cams (Picamera2)
    csi_sel: List[int] = []
    if PICAM_AVAILABLE:
        detected = csis.detect()
        if CSI_INDEXES_ENV:
            # Parse env list (ints)
            try:
                desired = [int(x) for x in CSI_INDEXES_ENV]
            except Exception:
                desired = []
            csi_sel = [i for i in desired if i in detected]
        else:
            csi_sel = detected
        csis.start(csi_sel)
    else:
        print("[CSI] Picamera2 not available; skipping CSI start.", flush=True)

    # PWM watchdog + MotorController
    threading.Thread(target=pwm_watchdog, daemon=True).start()
    globals()["control"] = MotorController(MOTORS, pwm, pwm_lock)
    control.start()

def cleanup_and_exit(*_):
    _stop.set()
    try: control.stop()
    except Exception: pass
    try:
        for m in MOTORS:
            with pwm_lock:
                pwm.set_pwm(m["ch"], 0, m["mid"])
    except: pass
    try: usbs.stop_all()
    except: pass
    try: csis.stop_all()
    except: pass
    print("\n[EXIT] cleaned up.", flush=True)
    try: time.sleep(0.3)
    except: pass
    os._exit(0)

def main():
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    start_threads()
    print(f"[READY] http://{HOST}:{PORT}  (USB: /usb/stream/PRIMARY or /usb/stream/<idx>, CSI: /csi/<idx>.mjpg)", flush=True)
    app.run(host=HOST, port=PORT, threaded=True)

if __name__ == "__main__":
    main()
