#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ROV Dashboard (snapshot CSI/RTSP + primed buffers + decoupled control):
- USB cam stays MJPEG (fast).
- CSI & RTSP endpoints now return a single JPEG snapshot per request (cheap).
- CSI buffers are primed with a black JPEG so UI shows immediately.
- libcamera-jpeg hardened with -n/--immediate and longer timeout.
"""

import os, sys, time, signal, subprocess, glob, math, shutil, io
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from flask import Flask, jsonify, Response, render_template_string, request, make_response
import cv2
import numpy as np

print("[BOOT] reef_dash starting...", flush=True)

# ------------------ GLOBAL CONFIG ------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

# UI refresh cadence
CAM_SWAP_SEC = float(os.environ.get("CAM_SWAP_SEC", "5.0"))         # RTSP card refresh
RTSP_SNAPSHOT_SEC = float(os.environ.get("RTSP_SNAPSHOT_SEC", "5.0"))
RTSP_MAX_PAR = int(os.environ.get("RTSP_MAX_PAR", "1"))

CSI_SNAPSHOT_SEC = float(os.environ.get("CSI_SNAPSHOT_SEC", "1.0"))
CSI_INDEXES = [i.strip() for i in os.environ.get("CSI_INDEXES", "0,1").split(",") if i.strip() != ""]

KNOWN_URLS = [
    "rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/101",
    "rtsp://admin:admin@192.168.0.108:554/h264/ch1/main/av_stream",
]
HUNT_TARGET = os.environ.get("HUNT_TARGET", "").strip()

USE_PWM = True
try:
    import smbus2
except Exception:
    USE_PWM = False

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|rw_timeout;1200000|stimeout;1200000"
)

MOTORS = [
    {"ch": 0, "mid": 400, "nudge": 30, "pos": 400},  # X
    {"ch": 1, "mid": 400, "nudge": 30, "pos": 400},  # Y
    {"ch": 2, "mid": 400, "nudge": 30, "pos": 400},  # Z
]
CLAMP = (0, 4095)
CMD_MAP: Dict[str, Optional[tuple]] = {
    "w": (0, +1), "s": (0, -1),
    "a": (1, +1), "d": (1, -1),
    "q": (2, +1), "e": (2, -1),
    "stop": None,
}

# ------------------ APP STATE ------------------
app = Flask(__name__)
_stop = threading.Event()

# ------------------ UTIL: FrameBuffer ------------------
class FrameBuffer:
    def __init__(self, prime_black: bool = False, w: int = 320, h: int = 180):
        self._buf: Optional[bytes] = None
        self._cond = threading.Condition()
        if prime_black:
            self._buf = self._make_black(w, h)

    def _make_black(self, w, h) -> bytes:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return jpg.tobytes() if ok else b""

    def set(self, b: bytes):
        with self._cond:
            self._buf = b
            self._cond.notify_all()

    def get(self, wait: bool = True) -> Optional[bytes]:
        with self._cond:
            if not wait:
                return self._buf
            while self._buf is None:
                self._cond.wait()
            return self._buf

# ------------------ USB CAMERA ------------------
_usb_label = "<none>"
_usb_buf = FrameBuffer(prime_black=True, w=FRAME_W, h=FRAME_H)

def discover_by_id_nodes() -> List[str]:
    nodes = sorted(
        p for p in glob.glob("/dev/v4l/by-id/*-video-index0")
        if os.path.islink(p) or os.path.exists(p)
    )
    if nodes:
        return [os.path.realpath(p) for p in nodes]
    return [p for p in ["/dev/video0","/dev/video1","/dev/video2","/dev/video3"] if os.path.exists(p)]

def open_uvc(node: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
        print("[USB] No video devices found", flush=True); return
    cap = None
    for n in nodes:
        c = open_uvc(n)
        if c:
            cap = c
            _usb_label = os.path.basename(n)
            print(f"[USB] opened {n}", flush=True)
            break
    if cap is None:
        print("[USB] No cameras could be opened", flush=True); return

    font = cv2.FONT_HERSHEY_SIMPLEX
    while not _stop.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02); continue
        if FRAME_W and FRAME_H:
            h, w = frame.shape[:2]
            if abs(w-FRAME_W) > 8 or abs(h-FRAME_H) > 8:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        try:
            cv2.putText(frame, f"USB: {_usb_label}", (12, 28), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
        except Exception:
            pass
        ok, jpg = cv2.imencode(".jpg", frame, encode_params)
        if ok: _usb_buf.set(jpg.tobytes())
    try: cap.release()
    except: pass
    print("[USB] worker stopped", flush=True)

# ------------------ PWM / CONTROLLER ------------------
class PWM:
    def __init__(self, addr=0x40, bus_id=1):
        self.ok = False
        self.addr = addr
        self.bus = None
        if not USE_PWM:
            print("[PWM] smbus2 not available; PWM disabled", flush=True); return
        try:
            self.bus = smbus2.SMBus(bus_id)
            MODE1, PRESCALE = 0x00, 0xFE
            self.bus.write_byte_data(addr, MODE1, 0x00); time.sleep(0.005)
            prescale_val = int(25000000.0 / (4096 * 50) - 1)  # 50 Hz
            self.bus.write_byte_data(addr, MODE1, 0x10)
            self.bus.write_byte_data(addr, PRESCALE, prescale_val)
            self.bus.write_byte_data(addr, MODE1, 0x00); time.sleep(0.005)
            self.bus.write_byte_data(addr, MODE1, 0xA1)
            self.ok = True; print("[PWM] initialized OK", flush=True)
        except Exception as e:
            print(f"[PWM] Disabled (I2C error): {e}", flush=True); self.ok = False

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
            self.ok = False; print(f"[PWM] write error: {e}", flush=True)

pwm_lock = threading.Lock()
pwm = PWM()
for m in MOTORS:
    m["pos"] = m["mid"]
    with pwm_lock: pwm.set_pwm(m["ch"], 0, m["mid"])

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
                addr = pwm.addr; globals()["pwm"] = PWM(addr=addr)
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
                out = cur if cur == tgt else (min(cur + self.auto_cfg.slew_per_tick, tgt) if tgt > cur else max(cur - self.auto_cfg.slew_per_tick, tgt))
                m["pos"] = out
                with self.lock: self.pwm.set_pwm(m["ch"], 0, int(out))
            time.sleep(max(0.0, dt - (time.time() - start)))

# ------------------ RTSP (snapshot workers) ------------------
rtsp_sem = threading.Semaphore(max(1, RTSP_MAX_PAR))

class CamWorker(threading.Thread):
    def __init__(self, idx: int, url: str, title: Optional[str] = None):
        super().__init__(daemon=True)
        self.idx = idx; self.url = url; self.title = title or url
        self.buf = FrameBuffer(prime_black=True, w=FRAME_W, h=FRAME_H)
        self._stop = threading.Event(); self.last_ok = 0.0

    def stop(self): self._stop.set()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        return cap if cap.isOpened() else None

    def _grab_one_frame(self) -> Optional[bytes]:
        cap = self._open_capture()
        if not cap: return None
        try:
            if FRAME_W: cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            if FRAME_H: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            if FRAME_FPS: cap.set(cv2.CAP_PROP_FPS, 5)
            ok, frame = cap.read()
            if not ok or frame is None: return None
            if FRAME_W and FRAME_H:
                h, w = frame.shape[:2]
                if abs(w - FRAME_W) > 8 or abs(h - FRAME_H) > 8:
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            return jpg.tobytes() if ok2 else None
        finally:
            try: cap.release()
            except: pass

    def run(self):
        try: os.nice(10)
        except Exception: pass
        print(f"[CAM{self.idx}] RTSP snapshot worker for {self.url}", flush=True)
        while not self._stop.is_set():
            start = time.time()
            if rtsp_sem.acquire(blocking=False):
                try:
                    jpg = self._grab_one_frame()
                    if jpg:
                        self.buf.set(jpg); self.last_ok = time.time()
                finally:
                    rtsp_sem.release()
            to_sleep = max(0.1, RTSP_SNAPSHOT_SEC - (time.time() - start))
            for _ in range(int(to_sleep * 10)):
                if self._stop.is_set(): break
                time.sleep(0.1)
        print(f"[CAM{self.idx}] stopped", flush=True)

class CamRegistry:
    def __init__(self):
        self._lock = threading.Lock(); self._cams: Dict[int, CamWorker] = {}; self._next_idx = 0
    def add(self, url: str, title: Optional[str] = None) -> int:
        with self._lock:
            for i, c in self._cams.items():
                if c.url == url: return i
            idx = self._next_idx; self._next_idx += 1
            w = CamWorker(idx, url, title or f"RTSP {idx}")
            self._cams[idx] = w; w.start()
            print(f"[REG] added RTSP cam {idx}: {url}", flush=True)
            return idx
    def list(self) -> List[Tuple[int, str, str]]:
        with self._lock: return [(i, c.url, c.title) for i, c in sorted(self._cams.items())]
    def get(self, idx: int) -> Optional[CamWorker]:
        with self._lock: return self._cams.get(idx)
    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

cams = CamRegistry()

# ------------------ CSI (snapshot workers with primed buffers) ------------------
LIBCAMERA_JPEG = shutil.which("libcamera-jpeg")

class CSICamWorker(threading.Thread):
    def __init__(self, idx: int, cam_id: str, title: Optional[str] = None):
        super().__init__(daemon=True)
        self.idx = idx; self.cam_id = cam_id; self.title = title or f"CSI {cam_id}"
        self.buf = FrameBuffer(prime_black=True, w=FRAME_W, h=FRAME_H)
        self._stop = threading.Event(); self.last_ok = 0.0

    def stop(self): self._stop.set()

    def _grab_with_libcamera(self) -> Optional[bytes]:
        if not LIBCAMERA_JPEG: return None
        # -n: no preview, --immediate: capture ASAP, -o - : stdout
        args = [
            LIBCAMERA_JPEG, "-n",
            "--immediate",
            "--camera", str(self.cam_id),
            "--width", str(FRAME_W), "--height", str(FRAME_H),
            "--quality", str(JPEG_QUALITY),
            "-o", "-"
        ]
        try:
            p = subprocess.run(args, timeout=5.0, capture_output=True)
            if p.returncode == 0 and p.stdout:
                return bytes(p.stdout)
            else:
                if p.stderr:
                    sys.stderr.write(f"[CSI{self.idx}] libcamera-jpeg stderr: {p.stderr.decode(errors='ignore')}\n")
        except Exception as e:
            sys.stderr.write(f"[CSI{self.idx}] libcamera-jpeg error: {e}\n")
        return None

    def _grab_with_v4l2(self) -> Optional[bytes]:
        csi_nodes = sorted(glob.glob("/dev/v4l/by-path/*csi*-video-index0"))
        nodes = [os.path.realpath(n) for n in csi_nodes] if csi_nodes else []
        if not nodes: return None
        node = nodes[int(self.cam_id) % len(nodes)]
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        if not cap.isOpened():
            try: cap.release()
            except: pass
            return None
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS, min(FRAME_FPS, 15))
            ok, frame = cap.read()
            if not ok or frame is None: return None
            h, w = frame.shape[:2]
            if abs(w - FRAME_W) > 8 or abs(h - FRAME_H) > 8:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            return jpg.tobytes() if ok2 else None
        finally:
            try: cap.release()
            except: pass

    def run(self):
        print(f"[CSI{self.idx}] snapshot worker for camera {self.cam_id}", flush=True)
        while not self._stop.is_set():
            start = time.time()
            jpg = self._grab_with_libcamera()
            if not jpg: jpg = self._grab_with_v4l2()
            if jpg:
                self.buf.set(jpg); self.last_ok = time.time()
            to_sleep = max(0.1, CSI_SNAPSHOT_SEC - (time.time() - start))
            for _ in range(int(to_sleep * 10)):
                if self._stop.is_set(): break
                time.sleep(0.1)
        print(f"[CSI{self.idx}] stopped", flush=True)

class CSIRegistry:
    def __init__(self):
        self._lock = threading.Lock(); self._cams: Dict[int, CSICamWorker] = {}; self._next_idx = 0
    def add(self, cam_id: str, title: Optional[str] = None) -> int:
        with self._lock:
            idx = self._next_idx; self._next_idx += 1
            w = CSICamWorker(idx, cam_id, title or f"CSI {cam_id}")
            self._cams[idx] = w; w.start()
            print(f"[REG] added CSI cam {idx} (id={cam_id})", flush=True)
            return idx
    def list(self) -> List[Tuple[int, str, str]]:
        with self._lock: return [(i, c.cam_id, c.title) for i, c in sorted(self._cams.items())]
    def get(self, idx: int) -> Optional[CSICamWorker]:
        with self._lock: return self._cams.get(idx)
    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

csis = CSIRegistry()

# ------------------ RTSP hunter (optional) ------------------
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
    args = ["ffprobe","-v","error","-hide_banner","-rtsp_transport",transport,
            "-select_streams","v:0","-show_entries","stream=codec_type",
            "-of","default=nokey=1:noprint_wrappers=1", url]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        return (out.returncode == 0 and "video" in out.stdout)
    except Exception:
        return False

def hunter_thread(hostport: str):
    print(f"[HUNT] starting for {hostport}", flush=True)
    host, _, port = hostport.partition(":"); port = port or "554"
    tried = 0; candidates: List[str] = []
    for u,pw in HUNT_CREDS:
        for path in HUNT_PATHS:
            candidates.append(f"rtsp://{u}:{pw}@{host}:{port}{path}")
    for path in HUNT_PATHS:
        candidates.append(f"rtsp://{host}:{port}{path}")
    for url in candidates:
        tried += 1
        if ffprobe_video(url, "udp", 2.5) or ffprobe_video(url, "tcp", 2.5):
            cams.add(url, f"Hunted @ {hostport}"); print(f"[HUNT] FOUND {url}", flush=True); return
        if tried % 10 == 0: print(f"[HUNT] tried {tried} candidates...", flush=True)
    print("[HUNT] done; none found", flush=True)

# ------------------ UI / PAGES ------------------
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
    <span class="pill" id="csis">CSI: ?</span>
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
        Health: <a href="/health" target="_blank">/health</a>.
        RTSP snapshots every {{swap}}s. CSI snapshots every {{csi_sec}}s.
      </small>
      <pre id="status"></pre>
    </section>

    <section>
      <h3>CSI Cameras</h3>
      <div class="row" id="csigrid"></div>
    </section>

    <section>
      <h3>IP Cameras (refresh every {{swap}}s)</h3>
      <div class="row" id="iprow"></div>
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
let csis = [];
async function loadCams(){
  try{
    const r = await fetch('/cams');
    const j = await r.json();
    cams = j.items || [];
    const row = document.getElementById('iprow');
    row.innerHTML = '';
    cams.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <h2>${c.title}</h2>
        <div class="sub">${c.url}</div>
        <img id="rtsp_${c.idx}" src="/cam/${c.idx}?t=${Date.now()}" />
      `;
      row.appendChild(card);
    });
  }catch(e){}
}
async function loadCSIs(){
  try{
    const r = await fetch('/csis');
    const j = await r.json();
    csis = j.items || [];
    const row = document.getElementById('csigrid');
    row.innerHTML = '';
    csis.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <h2>${c.title}</h2>
        <div class="sub">camera id: ${c.cam_id}</div>
        <img id="csi_${c.idx}" src="/csi/${c.idx}?t=${Date.now()}" />
      `;
      row.appendChild(card);
    });
  }catch(e){}
}

loadCams();
loadCSIs();

// Snapshot refresh timers (cheap, independent HTTP requests)
setInterval(()=>{
  cams.forEach(c=>{
    const img = document.getElementById('rtsp_'+c.idx);
    if(img){ img.src = '/cam/'+c.idx+'?t='+(Date.now()); }
  });
}, {{swap_ms}});

setInterval(()=>{
  csis.forEach(c=>{
    const img = document.getElementById('csi_'+c.idx);
    if(img){ img.src = '/csi/'+c.idx+'?t='+(Date.now()); }
  });
}, {{csi_ms}});

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
        swap_ms=int(CAM_SWAP_SEC*1000),
        csi_sec=CSI_SNAPSHOT_SEC,
        csi_ms=int(CSI_SNAPSHOT_SEC*1000),
    )

# USB stays MJPEG (fast continuous)
@app.route("/usb")
def usb_stream():
    def gen():
        boundary = b"--frame\r\n"
        while not _stop.is_set():
            frame = _usb_buf.get()
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# RTSP: single snapshot response
@app.route("/cam/<int:idx>")
def snapshot_cam(idx: int):
    worker = cams.get(idx)
    if worker is None:
        return "Camera not found", 404
    frame = worker.buf.get(wait=False)  # return latest, even if old
    if not frame:
        return "No frame yet", 503
    resp = make_response(frame)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store"
    return resp

# CSI: single snapshot response
@app.route("/csi/<int:idx>")
def snapshot_csi(idx: int):
    worker = csis.get(idx)
    if worker is None:
        return "CSI not found", 404
    frame = worker.buf.get(wait=False)
    if not frame:
        return "No frame yet", 503
    resp = make_response(frame)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/cams")
def list_cams():
    items = [{"idx": i, "url": u, "title": t} for i, u, t in cams.list()]
    return jsonify({"items": items, "count": len(items)})

@app.route("/csis")
def list_csis():
    items = [{"idx": i, "cam_id": cid, "title": t} for i, cid, t in csis.list()]
    return jsonify({"items": items, "count": len(items)})

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
            kw[k] = j[k]
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
        "rtsp_count": len(cams.list()),
        "csi_count": len(csis.list()),
        "csi_snapshot_sec": CSI_SNAPSHOT_SEC
    })

@app.route("/add", methods=["POST"])
def add_cam():
    j = request.get_json(force=True, silent=True) or {}
    url = j.get("url"); title = j.get("title")
    if not url: return jsonify({"ok": False, "error": "missing url"}), 400
    idx = cams.add(url, title)
    return jsonify({"ok": True, "idx": idx})

# ------------------ LIFECYCLE ------------------
def start_threads():
    threading.Thread(target=usb_worker, daemon=True).start()
    threading.Thread(target=pwm_watchdog, daemon=True).start()
    globals()["control"] = MotorController(MOTORS, pwm, pwm_lock); control.start()

def start_rtsp():
    for url in KNOWN_URLS: cams.add(url, f"Fixed: {url}")
    if HUNT_TARGET:
        threading.Thread(target=hunter_thread, args=(HUNT_TARGET,), daemon=True).start()
        print(f"[MAIN] hunter launched for {HUNT_TARGET}", flush=True)
    else:
        print("[MAIN] no hunter target configured.", flush=True)

def start_csi():
    if not CSI_INDEXES:
        print("[CSI] no CSI indexes configured (set CSI_INDEXES)", flush=True); return
    for cam_id in CSI_INDEXES:
        csis.add(cam_id, f"CSI Camera {cam_id}")

def cleanup_and_exit(*_):
    _stop.set()
    try: control.stop()
    except Exception: pass
    try:
        for m in MOTORS:
            with pwm_lock: pwm.set_pwm(m["ch"], 0, m["mid"])
    except: pass
    try: cams.stop_all()
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
    # start_rtsp()
    # start_csi()
    start_threads()
    print(f"[READY] http://{HOST}:{PORT}  (USB at /usb, RTSP snapshot at /cam/<idx>, CSI snapshot at /csi/<idx>)", flush=True)
    app.run(host=HOST, port=PORT, threaded=True)

if __name__ == "__main__":
    main()
