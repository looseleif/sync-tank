#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synctankone — SEE SEA TV (display-first, lean)
- Hero: USB MJPEG stream (left)
- Sidebar: CSI (Picamera2 MJPEG, like synctankthree) + RTSP snapshots (~5s)
- PWM (PCA9685) watchdog + AUTO motor loop (status-only UI)
- Minimal CPU: only hero is continuous; RTSP tiles are snapshots; CSI is Picamera2 MJPEG (as per synctankthree request)
- Route aliases included: /csi/<idx>.mjpg and /csi/<idx>; /rtsp/<idx>.jpg and /cam/<idx>
"""

import os, time, signal, glob, io, math, threading, subprocess, shutil
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from flask import Flask, jsonify, Response, render_template_string, abort, make_response
import cv2
import numpy as np

# ---------------- ENV / CONFIG ----------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

# USB hero camera defaults
FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
MAX_USB_CAMS = int(os.environ.get("MAX_USB_CAMS", "4"))
USB_PRIMARY = int(os.environ.get("USB_PRIMARY", "0"))

# CSI (Picamera2 workers — like synctankthree)
CSI_W = int(os.environ.get("CSI_W", "1280"))
CSI_H = int(os.environ.get("CSI_H", "720"))
CSI_FPS = int(os.environ.get("CSI_FPS", "12"))  # keep gentle
CSI_JPEG_QUALITY = int(os.environ.get("CSI_JPEG_QUALITY", "70"))
CSI_INDEXES_ENV = [s.strip() for s in os.environ.get("CSI_INDEXES", "0,1").split(",") if s.strip()]

# RTSP lightweight snapshots (serialized; ~5s default)
RTSP_SNAPSHOT_SEC = float(os.environ.get("RTSP_SNAPSHOT_SEC", "5.0"))
RTSP_MAX_PAR = int(os.environ.get("RTSP_MAX_PAR", "1"))  # serialize grabs to avoid spikes
KNOWN_URLS = [
    "rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/101",
    "rtsp://admin:admin@192.168.0.108:554/h264/ch1/main/av_stream",
]
HUNT_TARGET = os.environ.get("HUNT_TARGET", "").strip()

# PWM
USE_PWM_ENV = os.environ.get("USE_PWM", "1").lower() not in ("0","false","no")

# RTSP CV/FFmpeg hints
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|rw_timeout;1200000|stimeout;1200000")

# ---------------- THEME ----------------
THEME_TITLE = "synctankone"
THEME_NAME  = "SEE SEA TV"
ACCENT = "#6EE7FF"
BG     = "#07090b"
PANEL  = "#0e1216"
BORD   = "#1a2026"
FG     = "#e8f0f6"
SUBFG  = "#93a3b4"

print("[BOOT] synctankone SEE SEA TV starting...", flush=True)

app = Flask(__name__)
_stop = threading.Event()

# ---------------- UTIL ----------------
class FrameBuffer:
    def __init__(self, prime_black=False, w=320, h=180, q=60):
        self._buf: Optional[bytes] = None
        self._cond = threading.Condition()
        if prime_black:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            self._buf = jpg.tobytes() if ok else None
    def set(self, b: bytes):
        with self._cond:
            self._buf = b
            self._cond.notify_all()
    def get(self) -> bytes:
        with self._cond:
            while self._buf is None:
                self._cond.wait()
            return self._buf
    def peek(self) -> Optional[bytes]:
        return self._buf

# ---------------- USB CAMS (hero + tiles) ----------------
def _by_path_usb_nodes() -> List[str]:
    return [os.path.realpath(p) for p in sorted(glob.glob("/dev/v4l/by-path/*-usb-*-video-index0")) if os.path.exists(p)]

def discover_usb_nodes(max_n: int) -> List[str]:
    nodes = _by_path_usb_nodes()
    if not nodes:
        # best-effort fallback
        nodes = [f"/dev/video{i}" for i in range(8) if os.path.exists(f"/dev/video{i}")]
    uniq, seen = [], set()
    for n in nodes:
        if n not in seen:
            uniq.append(n); seen.add(n)
        if len(uniq) >= max_n: break
    return uniq

class USBWorker(threading.Thread):
    def __init__(self, idx: int, node: str):
        super().__init__(daemon=True)
        self.idx, self.node = idx, node
        self.buf = FrameBuffer(prime_black=True, w=FRAME_W, h=FRAME_H, q=JPEG_QUALITY)
        self._stop = threading.Event()
        self.ok = False
        self.last_error = ""
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
            except: pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
            try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except: pass
        except Exception as e:
            self.last_error = f"cfg: {e}"
        return cap
    def run(self):
        print(f"[USB{self.idx}] opening {self.node}", flush=True)
        cap = None
        for _ in range(3):
            cap = self._open()
            if cap: break
            time.sleep(0.25)
        if not cap: return
        # warm-up
        for _ in range(5):
            ok, _ = cap.read()
            if self._stop.is_set() or ok: break
            time.sleep(0.02)
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        self.ok = True
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02); continue
            h, w = frame.shape[:2]
            if abs(w-FRAME_W) > 8 or abs(h-FRAME_H) > 8:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            ok2, jpg = cv2.imencode(".jpg", frame, enc)
            if ok2: self.buf.set(jpg.tobytes())
        try: cap.release()
        except: pass
        print(f"[USB{self.idx}] stopped", flush=True)

class USBRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._cams: Dict[int, USBWorker] = {}
        self._nodes: List[str] = []
        self._primary_idx = USB_PRIMARY
    def start(self, max_n: int):
        with self._lock:
            self._nodes = discover_usb_nodes(max_n)
            if not self._nodes:
                print("[USB] no cameras discovered", flush=True)
            for idx, node in enumerate(self._nodes):
                w = USBWorker(idx, node); self._cams[idx] = w; w.start()
                print(f"[REG] USB {idx}: {node}", flush=True)
            if self._primary_idx >= len(self._nodes):
                self._primary_idx = 0
    def list(self) -> List[Tuple[int,str]]:
        with self._lock:
            return [(i, self._nodes[i]) for i in sorted(self._cams.keys())]
    def get(self, idx: int) -> Optional[USBWorker]:
        with self._lock:
            return self._cams.get(idx)
    def primary_idx(self) -> int:
        with self._lock:
            return self._primary_idx
    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

usbs = USBRegistry()

# ---------------- CSI CAMS (Picamera2 like synctankthree) ----------------
try:
    from picamera2 import Picamera2
    from PIL import Image
    PICAM_AVAILABLE = True
except Exception:
    PICAM_AVAILABLE = False

class SimpleCSICam:
    def __init__(self, cam_index:int, size=(1280,720), fps:int=12, q:int=70):
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
            self.last_error="picamera2 missing"; return
        try:
            self.picam = Picamera2(camera_num=self.idx)
            cfg = self.picam.create_preview_configuration(main={"size": self.size, "format": "RGB888"})
            self.picam.configure(cfg); self.picam.start(); time.sleep(0.3)
            self._stop.clear()
            self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
            self.ok = True
            print(f"[CSI{self.idx}] started", flush=True)
        except Exception as e:
            self.last_error = f"start: {e}"
            try:
                if self.picam: self.picam.close()
            except: pass
            self.picam = None
    def stop(self):
        self._stop.set()
        if self._t: self._t.join(timeout=1.0)
        if self.picam:
            try: self.picam.stop()
            except: pass
            try: self.picam.close()
            except: pass
        self.picam = None; self._t=None; self.ok=False
    def _loop(self):
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                arr = self.picam.capture_array("main")
                with io.BytesIO() as buff:
                    Image.fromarray(arr).save(buff, format="JPEG", quality=self.q, optimize=True)
                    self.latest_jpeg = buff.getvalue()
            except Exception as e:
                self.last_error = f"cap: {e}"
            dt = time.time() - t0
            if dt < period: time.sleep(period - dt)
    def frames(self):
        boundary = b"--frame\r\n"
        while not _stop.is_set():
            if self.latest_jpeg is None: time.sleep(0.05); continue
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
        if not PICAM_AVAILABLE: self._info=[]; return []
        try:
            info = Picamera2.global_camera_info()
            # Keep strings for /health display
            self._info = [str(x) for x in info]
            return list(range(len(info)))
        except Exception:
            self._info=[]; return []
    def start(self, indexes: Optional[List[int]] = None):
        with self._lock:
            avail = self.detect()
            # If indexes are supplied, use them; else start all available
            use = []
            if indexes:
                for i in indexes:
                    try:
                        ii = int(i)
                        if ii in avail: use.append(ii)
                    except: pass
            else:
                use = avail
            for i in use:
                if i not in self._cams:
                    cam = SimpleCSICam(i, size=(CSI_W, CSI_H), fps=CSI_FPS, q=CSI_JPEG_QUALITY)
                    self._cams[i] = cam; cam.start()
                    print(f"[REG] CSI {i}", flush=True)
            if not use: print("[CSI] none to start", flush=True)
    def list(self) -> List[Tuple[int,str]]:
        with self._lock:
            avail = self.detect()
            out=[]
            for i in sorted(self._cams.keys()):
                desc = self._info[i] if (self._info and i < len(self._info)) else ""
                if i in avail:
                    out.append((i, desc))
            return out
    def get(self, idx:int) -> Optional[SimpleCSICam]:
        with self._lock:
            return self._cams.get(idx)
    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

csis = CSIRegistry()

# ---------------- RTSP SNAPSHOT WORKERS (lightweight) ----------------
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|rw_timeout;1200000|stimeout;1200000")
rtsp_sem = threading.Semaphore(max(1, RTSP_MAX_PAR))

class RTSPWorker(threading.Thread):
    def __init__(self, idx:int, url:str, title:str):
        super().__init__(daemon=True)
        self.idx=idx; self.url=url; self.title=title
        self.buf=FrameBuffer(prime_black=True, w=640, h=360, q=70)
        self._stop=threading.Event()
    def stop(self): self._stop.set()
    def _open(self)->Optional[cv2.VideoCapture]:
        cap=cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        except: pass
        return cap if cap.isOpened() else None
    def _grab(self)->Optional[bytes]:
        cap=self._open()
        if not cap: return None
        try:
            try: cap.set(cv2.CAP_PROP_FPS, 5)
            except: pass
            ok, frame = cap.read()
            if not ok or frame is None: return None
            h,w=frame.shape[:2]
            if w>640 or h>360:
                frame=cv2.resize(frame,(640,360),cv2.INTER_AREA)
            ok2,jpg=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),70])
            return jpg.tobytes() if ok2 else None
        finally:
            try: cap.release()
            except: pass
    def run(self):
        try: os.nice(10)
        except: pass
        print(f"[RTSP{self.idx}] {self.url}", flush=True)
        while not self._stop.is_set():
            t0=time.time()
            if rtsp_sem.acquire(blocking=False):
                try:
                    jpg=self._grab()
                    if jpg: self.buf.set(jpg)
                finally:
                    rtsp_sem.release()
            sleep=max(0.1, RTSP_SNAPSHOT_SEC-(time.time()-t0))
            for _ in range(int(sleep*10)):
                if self._stop.is_set(): break
                time.sleep(0.1)
        print(f"[RTSP{self.idx}] stopped", flush=True)

class RTSPRegistry:
    def __init__(self):
        self._lock=threading.Lock()
        self._cams:Dict[int,RTSPWorker]={}
        self._next=0
    def add(self,url:str,title:str)->int:
        with self._lock:
            for i,c in self._cams.items():
                if c.url==url: return i
            idx=self._next; self._next+=1
            w=RTSPWorker(idx,url,title); self._cams[idx]=w; w.start(); return idx
    def list(self)->List[Tuple[int,str,str]]:
        with self._lock:
            return [(i,c.url,c.title) for i,c in sorted(self._cams.items())]
    def get(self,idx:int)->Optional[RTSPWorker]:
        with self._lock: return self._cams.get(idx)
    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

rtsp = RTSPRegistry()

# Optional RTSP hunter (best-effort)
HUNT_PATHS=[
    "/h264/ch1/main/av_stream","/live/ch00_0","/live","/videoMain",
    "/Streaming/Channels/101","/h264Preview_01_main",
    "/cam/realmonitor?channel=1&subtype=0","/cam/realmonitor?channel=1&subtype=1",
]
HUNT_CREDS=[("admin","admin"),("admin","123456"),("admin","")]

def _ffprobe_ok(url:str,transport="tcp",timeout_s=3.0)->bool:
    args=["ffprobe","-v","error","-hide_banner","-rtsp_transport",transport,"-select_streams","v:0",
          "-show_entries","stream=codec_type","-of","default=nokey=1:noprint_wrappers=1",url]
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=timeout_s)
        return r.returncode==0 and "video" in r.stdout
    except: return False

def hunter(hostport:str):
    print(f"[HUNT] {hostport}", flush=True)
    host,_,port=hostport.partition(":"); port=port or "554"
    cands=[f"rtsp://{u}:{pw}@{host}:{port}{p}" for u,pw in HUNT_CREDS for p in HUNT_PATHS]+\
          [f"rtsp://{host}:{port}{p}" for p in HUNT_PATHS]
    for i,url in enumerate(cands,1):
        if _ffprobe_ok(url,"udp",2.0) or _ffprobe_ok(url,"tcp",2.5):
            rtsp.add(url,f"Hunted {hostport}"); print(f"[HUNT] FOUND {url}", flush=True); return
        if i%10==0: print(f"[HUNT] tried {i}...", flush=True)
    print("[HUNT] none", flush=True)

# ---------------- PWM / MOTOR (AUTO only visible) ----------------
USE_PWM = USE_PWM_ENV
try:
    import smbus2
except Exception:
    USE_PWM = False

MOTORS = [
    {"ch": 0, "mid": 400, "pos": 400},  # X
    {"ch": 4, "mid": 400, "pos": 400},  # Y
    {"ch": 8, "mid": 400, "pos": 400},  # Z
]
CLAMP = (0, 4095)

class PWM:
    def __init__(self, addr=0x40, bus_id=1):
        self.ok=False; self.addr=addr; self.bus=None
        if not USE_PWM:
            print("[PWM] disabled", flush=True); return
        try:
            self.bus = smbus2.SMBus(bus_id)
            MODE1, PRESCALE = 0x00, 0xFE
            self.bus.write_byte_data(addr, MODE1, 0x00); time.sleep(0.005)
            prescale_val = int(25000000.0 / (4096 * 50) - 1)
            self.bus.write_byte_data(addr, MODE1, 0x10)
            self.bus.write_byte_data(addr, PRESCALE, prescale_val)
            self.bus.write_byte_data(addr, MODE1, 0x00); time.sleep(0.005)
            self.bus.write_byte_data(addr, MODE1, 0xA1)
            self.ok=True; print("[PWM] OK", flush=True)
        except Exception as e:
            print(f"[PWM] I2C error: {e}", flush=True)
    def set_pwm(self, ch:int, on:int, off:int):
        if not self.ok: return
        off = max(CLAMP[0], min(CLAMP[1], off))
        base = 0x06 + 4 * ch
        try:
            self.bus.write_byte_data(self.addr, base+0, on & 0xFF)
            self.bus.write_byte_data(self.addr, base+1, (on>>8) & 0xFF)
            self.bus.write_byte_data(self.addr, base+2, off & 0xFF)
            self.bus.write_byte_data(self.addr, base+3, (off>>8) & 0xFF)
        except Exception as e:
            self.ok=False; print(f"[PWM] write fail: {e}", flush=True)

pwm_lock = threading.Lock()
pwm = PWM()
for m in MOTORS:
    m["pos"] = m["mid"]
    with pwm_lock:
        pwm.set_pwm(m["ch"], 0, m["mid"])

def pwm_watchdog():
    MODE1 = 0x00
    while not _stop.is_set():
        time.sleep(3.0)
        if not USE_PWM: continue
        try:
            if pwm.bus is None: raise IOError("bus None")
            _ = pwm.bus.read_byte_data(pwm.addr, MODE1)
            pwm.ok = True
        except Exception as e:
            print(f"[PWM] watchdog reinit: {e}", flush=True)
            try:
                addr = pwm.addr
                globals()["pwm"] = PWM(addr=addr)
            except Exception as ee:
                print(f"[PWM] reinit failed: {ee}", flush=True)

@dataclass
class AutoConfig:
    radius_xy: int = 120
    radius_z: int  = 40
    period_s: float = 12.0
    z_period_s: float = 7.0
    slew_per_tick: int = 12
    tick_hz: float = 18.0
    enable_x: bool = True
    enable_y: bool = True
    enable_z: bool = False

class MotorController(threading.Thread):
    def __init__(self, motors, pwm, lock):
        super().__init__(daemon=True)
        self.motors, self.pwm, self.lock = motors, pwm, lock
        self._stop = threading.Event()
        self.mode = "auto"
        self.auto_cfg = AutoConfig()
        self._t0 = time.time()
        self.targets = [m["mid"] for m in motors]
    def stop(self): self._stop.set()
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
            self._compute_auto_targets(start)
            for i, m in enumerate(self.motors):
                cur, tgt = m["pos"], int(self.targets[i])
                step = self.auto_cfg.slew_per_tick
                out = cur if cur == tgt else (min(cur + step, tgt) if tgt > cur else max(cur - step, tgt))
                m["pos"] = out
                with self.lock: self.pwm.set_pwm(m["ch"], 0, int(out))
            slp = dt - (time.time() - start)
            if slp > 0: time.sleep(slp)

# ---------------- HTML ----------------
PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{title}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { color-scheme: dark; }
    html, body { height:100%; }
    body { margin:0; background:{{BG}}; color:{{FG}}; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; overflow:hidden; }
    header { height:46px; display:flex; align-items:center; gap:10px; padding:0 12px; background:{{PANEL}}; border-bottom:1px solid {{BORD}}; }
    .brand { font-weight:700; letter-spacing:.4px; color:{{ACCENT}}; }
    .stat { font-size:12px; color:{{SUBFG}}; }
    .wrap { height: calc(100vh - 46px); display:grid; grid-template-columns: 2fr 1fr; gap:8px; padding:8px; }
    .hero, .tile { background:#000; border:1px solid {{BORD}}; border-radius:10px; overflow:hidden; display:flex; align-items:center; justify-content:center; }
    .hero img, .tile img { width:100%; height:100%; object-fit:contain; background:#000; }
    .tiles { display:grid; grid-template-columns: 1fr 1fr; grid-auto-rows: minmax(0, 1fr); gap:8px; min-height:0; }
    .wm { position:fixed; right:10px; bottom:8px; color:{{SUBFG}}; opacity:0.25; font-size:12px; user-select:none; pointer-events:none; }
    a { color:#7fc7ff }
  </style>
</head>
<body>
  <header>
    <div class="brand">{{theme}}</div>
    <div class="stat" id="s_pwm">PWM: …</div>
    <div class="stat" id="s_mode">MODE: …</div>
    <div class="stat" id="s_usb">USB: …</div>
    <div class="stat" id="s_csi">CSI: …</div>
    <div class="stat" id="s_rtsp">RTSP: …</div>
    <div style="margin-left:auto;color:{{SUBFG}};">http://{{host}}:{{port}}</div>
  </header>

  <div class="wrap">
    <section class="hero">
      <!-- Hero: primary USB MJPEG -->
      <img id="hero" src="/usb/stream/PRIMARY" />
    </section>
    <section class="tiles" id="tiles"></section>
  </div>

  <div class="wm">leifberryone</div>

<script>
async function refreshStatus(){
  try{
    const r = await fetch('/health'); const j = await r.json();
    document.getElementById('s_pwm').textContent  = 'PWM: ' + (j.pwm_ok ? 'OK' : 'ERR');
    document.getElementById('s_mode').textContent = 'MODE: ' + (j.mode || 'auto');
    document.getElementById('s_usb').textContent  = 'USB: ' + j.usb_count + ' (primary ' + j.usb_primary + ')';
    document.getElementById('s_csi').textContent  = 'CSI: ' + j.csi_count;
    document.getElementById('s_rtsp').textContent = 'RTSP: ' + j.rtsp_count;
  }catch(e){}
}
setInterval(refreshStatus, 5000);
refreshStatus();

async function loadTiles(){
  const tiles = document.getElementById('tiles');
  tiles.innerHTML = '';
  // Show secondary USBs (if any)
  try{
    const ru = await fetch('/usbs'); const uj = await ru.json();
    (uj.items||[]).forEach(c=>{
      const card = document.createElement('div');
      card.className = 'tile';
      card.innerHTML = `<img src="/usb/stream/${c.idx}" />`;
      tiles.appendChild(card);
    });
  }catch(e){}
  // CSI (Picamera2 MJPEG)
  try{
    const rc = await fetch('/csis'); const cj = await rc.json();
    (cj.items||[]).forEach(c=>{
      const card = document.createElement('div');
      card.className = 'tile';
      // Prefer MJPEG, but also allow alias without .mjpg
      card.innerHTML = `<a href="/csi/${c.idx}.mjpg" target="_blank"><img src="/csi/${c.idx}.mjpg" /></a>`;
      tiles.appendChild(card);
    });
  }catch(e){}
  // RTSP snapshots (refresh via JS every 5s)
  try{
    const rr = await fetch('/cams'); const rj = await rr.json();
    (rj.items||[]).forEach(c=>{
      const card = document.createElement('div');
      card.className = 'tile';
      card.innerHTML = `<img id="rtsp_${c.idx}" src="/rtsp/${c.idx}.jpg?ts=${Date.now()}" />`;
      tiles.appendChild(card);
    });
    function tickRTSP(){
      document.querySelectorAll('[id^="rtsp_"]').forEach(img=>{
        img.src = img.src.split('?')[0] + '?ts=' + Date.now();
      });
      setTimeout(tickRTSP, {{rtsp_ms}});
    }
    tickRTSP();
  }catch(e){}
}
loadTiles();
</script>
</body>
</html>
"""

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template_string(
        PAGE,
        title=THEME_TITLE, theme=THEME_NAME,
        BG=BG, PANEL=PANEL, BORD=BORD, FG=FG, SUBFG=SUBFG, ACCENT=ACCENT,
        host=HOST, port=PORT, rtsp_ms=int(RTSP_SNAPSHOT_SEC*1000)
    )

# USB MJPEG (hero & tiles)
@app.route("/usb/stream/<sel>")
def usb_stream(sel: str):
    if sel.upper() == "PRIMARY":
        idx = usbs.primary_idx()
    else:
        try: idx = int(sel)
        except: return "bad selector", 400
    w = usbs.get(idx)
    if not w: return "USB not found", 404
    def gen():
        boundary = b"--frame\r\n"
        while not _stop.is_set():
            frame = w.buf.get()
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/usbs")
def list_usbs():
    items=[]
    for i, node in usbs.list():
        w = usbs.get(i)
        items.append({"idx": i, "node": node, "ok": bool(w and w.ok), "err": getattr(w, "last_error", "")})
    return jsonify({"items": items, "count": len(items)})

# CSI MJPEG (as in synctankthree) + alias without .mjpg
def _csi_stream(idx: int):
    cam = csis.get(idx)
    if not cam: abort(404, f"CSI {idx} not running")
    return Response(cam.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/csi/<int:idx>.mjpg")
def csi_mjpg(idx: int):
    detected = [i for i, _ in csis.list()]
    if idx not in detected: abort(404, f"CSI {idx} not available")
    return _csi_stream(idx)

@app.route("/csi/<int:idx>")
def csi_alias(idx: int):
    detected = [i for i, _ in csis.list()]
    if idx not in detected: abort(404, f"CSI {idx} not available")
    return _csi_stream(idx)

@app.route("/csis")
def list_csis():
    items=[]
    for i, desc in csis.list():
        w = csis.get(i)
        items.append({"idx": i, "desc": desc, "ok": bool(w and w.ok and w.latest_jpeg is not None), "err": getattr(w, "last_error","")})
    return jsonify({"items": items, "count": len(items)})

# RTSP: list + JPEG snapshot + /cam alias
@app.route("/cams")
def list_rtsp():
    items=[{"idx":i,"url":u,"title":t} for i,u,t in rtsp.list()]
    return jsonify({"items": items, "count": len(items)})

@app.route("/rtsp/<int:idx>.jpg")
def rtsp_jpg(idx:int):
    w=rtsp.get(idx)
    if not w: abort(404)
    b=w.buf.peek()
    if not b: abort(503, "no frame yet")
    r=make_response(b); r.headers["Content-Type"]="image/jpeg"; return r

# alias to match earlier /cam/<idx> requests
@app.route("/cam/<int:idx>")
def cam_alias(idx:int):
    return rtsp_jpg(idx)

# Health (status-only)
@app.route("/health")
def health():
    usb_items=[]
    for i, node in usbs.list():
        w = usbs.get(i)
        usb_items.append({"idx": i, "node": node, "ok": bool(w and w.ok), "err": getattr(w, "last_error", "")})
    csi_items=[]
    for i, desc in csis.list():
        w = csis.get(i)
        csi_items.append({"idx": i, "desc": desc, "ok": bool(w and w.ok and w.latest_jpeg is not None), "err": getattr(w, "last_error","")})
    return jsonify({
        "pwm_ok": bool(pwm.ok),
        "mode": "auto",
        "motors": [{"ch": m["ch"], "pos": int(m["pos"]), "mid": m["mid"]} for m in MOTORS],
        "targets": getattr(globals().get("control", None), "targets", []),
        "usb_count": len(usb_items),
        "usb_primary": usbs.primary_idx(),
        "csi_count": len(csi_items),
        "rtsp_count": len(rtsp.list()),
    })

# Add RTSP at runtime (optional)
@app.route("/add", methods=["POST"])
def add_rtsp():
    from flask import request
    j = request.get_json(force=True, silent=True) or {}
    url = j.get("url"); title = j.get("title") or url
    if not url: return jsonify({"ok":False,"err":"missing url"}),400
    idx=rtsp.add(url,title); return jsonify({"ok":True,"idx":idx})

# ---------------- LIFECYCLE ----------------
def start_threads():
    # USB
    usbs.start(MAX_USB_CAMS)
    # CSI (use provided indexes if any; else auto-detect)
    if PICAM_AVAILABLE:
        try:
            desired = [int(x) for x in CSI_INDEXES_ENV] if CSI_INDEXES_ENV else []
        except Exception:
            desired = []
        csis.start(desired if desired else None)
    else:
        print("[CSI] picamera2 not available; skipping", flush=True)
    # RTSP
    for url in KNOWN_URLS:
        rtsp.add(url, f"Fixed: {url}")
    if HUNT_TARGET:
        threading.Thread(target=hunter, args=(HUNT_TARGET,), daemon=True).start()
        print(f"[MAIN] hunter launched for {HUNT_TARGET}", flush=True)
    # PWM watchdog + motor loop
    threading.Thread(target=pwm_watchdog, daemon=True).start()
    globals()["control"] = MotorController(MOTORS, pwm, pwm_lock)
    control.start()

def cleanup_and_exit(*_):
    _stop.set()
    try: control.stop()
    except: pass
    try:
        for m in MOTORS:
            with pwm_lock: pwm.set_pwm(m["ch"], 0, m["mid"])
    except: pass
    try: usbs.stop_all()
    except: pass
    try: csis.stop_all()
    except: pass
    try: rtsp.stop_all()
    except: pass
    print("\n[EXIT] cleaned up.", flush=True)
    try: time.sleep(0.3)
    except: pass
    os._exit(0)

def main():
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    start_threads()
    print(f"[READY] http://{HOST}:{PORT}  (Hero USB: /usb/stream/PRIMARY | CSI: /csi/<idx>.mjpg | RTSP: /rtsp/<idx>.jpg)", flush=True)
    # Disable reloader for predictable threads
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False, debug=False)

if __name__ == "__main__":
    main()
