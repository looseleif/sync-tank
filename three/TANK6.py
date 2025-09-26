#!/usr/bin/env python3
# TANK — SYNC TANK Edition
# - AUTO starts enabled; Manual = hold mids (no per-axis nudges).
# - Threads for USB cams, CSI cams, PWM watchdog, Motor loop.
# - One-screen dashboard (landscape, no scrolling): Hero + side tiles.
# - Buttons fixed to POST; axis checkboxes update targets live.

import os, sys, time, signal, glob, io, math, threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from flask import Flask, jsonify, Response, render_template_string, request, abort
import cv2
import numpy as np

# ---------------- ENV / CONFIG ----------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

# USB camera defaults
FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
MAX_USB_CAMS = int(os.environ.get("MAX_USB_CAMS", "4"))   # more side tiles if you like
USB_PRIMARY = int(os.environ.get("USB_PRIMARY", "0"))

# CSI (optional)
CSI_W = int(os.environ.get("CSI_W", "1280"))
CSI_H = int(os.environ.get("CSI_H", "720"))
CSI_FPS = int(os.environ.get("CSI_FPS", "15"))
CSI_JPEG_QUALITY = int(os.environ.get("CSI_JPEG_QUALITY", "70"))
CSI_INDEXES_ENV = [s.strip() for s in os.environ.get("CSI_INDEXES", "0,1").split(",") if s.strip() != ""]

# PWM
USE_PWM_ENV = os.environ.get("USE_PWM", "1").lower() not in ("0","false","no")

# ---- CSI deps (optional) ----
try:
    from picamera2 import Picamera2
    from PIL import Image
    PICAM_AVAILABLE = True
except Exception:
    PICAM_AVAILABLE = False

# ---------------- THEME ----------------
# Edit these to re-skin the dashboard quickly.
THEME_NAME = "SYNC TANK"
ACCENT = "#6EE7FF"    # cyan accent
BG     = "#07090b"    # page background
PANEL  = "#0e1216"    # panel background
BORD   = "#1a2026"    # borders
FG     = "#e8f0f6"    # primary text
SUBFG  = "#93a3b4"    # secondary text

print("[BOOT] TANK2 (SYNC TANK) starting...", flush=True)

app = Flask(__name__)
_stop = threading.Event()

# ---------------- UTIL ----------------
class FrameBuffer:
    def __init__(self, prime_black=False, w=320, h=180, q=60, label=""):
        self._buf: Optional[bytes] = None
        self._cond = threading.Condition()
        if prime_black:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            if label:
                cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
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

# ---------------- USB CAMS ----------------
def _by_path_usb_nodes() -> List[str]:
    return [os.path.realpath(p) for p in sorted(glob.glob("/dev/v4l/by-path/*-usb-*-video-index0")) if os.path.exists(p)]

def _fallback_usb_nodes() -> List[str]:
    # Fallback to anything by-path thinks is USB
    byp = set(_by_path_usb_nodes())
    out = []
    for n in range(64):
        dev = f"/dev/video{n}"
        if os.path.exists(dev) and dev in byp:
            out.append(dev)
    return out

def discover_usb_nodes(max_n: int) -> List[str]:
    nodes = _by_path_usb_nodes()
    if not nodes:
        nodes = _fallback_usb_nodes()
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
        self.buf = FrameBuffer(prime_black=True, w=FRAME_W, h=FRAME_H, label=f"USB{idx}")
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

    def set_primary(self, idx: int) -> bool:
        with self._lock:
            if idx in self._cams: self._primary_idx = idx; return True
            return False

    def primary_idx(self) -> int:
        with self._lock:
            return self._primary_idx

    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

usbs = USBRegistry()

# ---------------- CSI CAMS (optional) ----------------
class SimpleCSICam:
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
        if not PICAM_AVAILABLE: self.last_error="picamera2 missing"; return
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
            self._info = [str(x) for x in info]
            return list(range(len(info)))
        except Exception:
            self._info=[]; return []

    def start(self, indexes: Optional[List[int]] = None):
        with self._lock:
            avail = self.detect()
            use = avail if not indexes else [i for i in indexes if i in avail]
            for i in use:
                if i not in self._cams:
                    cam = SimpleCSICam(i, size=(CSI_W, CSI_H), fps=CSI_FPS, q=CSI_JPEG_QUALITY)
                    self._cams[i] = cam; cam.start()
                    print(f"[REG] CSI {i}", flush=True)
            if not use: print("[CSI] none", flush=True)

    def list(self) -> List[Tuple[int,str]]:
        with self._lock:
            avail = self.detect()
            out=[]
            for i in sorted(self._cams.keys()):
                desc = self._info[i] if (self._info and i < len(self._info)) else ""
                if i in avail: out.append((i, desc))
            return out

    def get(self, idx:int) -> Optional[SimpleCSICam]:
        with self._lock:
            return self._cams.get(idx)

    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

csis = CSIRegistry()

# ---------------- PWM / MOTOR ----------------
USE_PWM = USE_PWM_ENV
try:
    import smbus2
except Exception:
    USE_PWM = False

MOTORS = [
    {"ch": 0, "mid": 400, "nudge": 30, "pos": 400},  # X
    {"ch": 4, "mid": 400, "nudge": 30, "pos": 400},  # Y
    {"ch": 8, "mid": 400, "nudge": 30, "pos": 400},  # Z
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
        time.sleep(2.0)
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
    tick_hz: float = 20.0
    enable_x: bool = True
    enable_y: bool = True
    enable_z: bool = False

class MotorController(threading.Thread):
    def __init__(self, motors, pwm, lock):
        super().__init__(daemon=True)
        self.motors, self.pwm, self.lock = motors, pwm, lock
        self._stop = threading.Event()
        self.mode = "auto"       # start in auto
        self.auto_cfg = AutoConfig()
        self._t0 = time.time()
        self.targets = [m["mid"] for m in motors]

    def stop(self): self._stop.set()

    def set_mode(self, mode: str):
        self.mode = "auto" if mode.lower().startswith("a") else "manual"
        if self.mode == "manual":
            self.targets = [m["mid"] for m in self.motors]

    def set_all_mid(self):
        for i, m in enumerate(self.motors): self.targets[i] = m["mid"]

    def set_auto_params(self, **kw):
        for k, v in kw.items():
            if hasattr(self.auto_cfg, k):
                if k in ("period_s", "z_period_s", "tick_hz"): setattr(self.auto_cfg, k, float(v))
                elif k.startswith("enable_"): setattr(self.auto_cfg, k, bool(v) if isinstance(v, bool) else (str(v).lower() in ("1","true","yes","on")))
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
                cur, tgt = m["pos"], int(self.targets[i])
                step = self.auto_cfg.slew_per_tick
                out = cur if cur == tgt else (min(cur + step, tgt) if tgt > cur else max(cur - step, tgt))
                m["pos"] = out
                with self.lock: self.pwm.set_pwm(m["ch"], 0, int(out))
            time.sleep(max(0.0, dt - (time.time() - start)))

# ---------------- HTML (SYNC TANK themed, no-scroll, landscape grid) ----------------
PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{theme_name}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { color-scheme: dark; }
    html, body { height:100%; }
    body { margin:0; background:{{BG}}; color:{{FG}}; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; overflow:hidden; }

    /* Top bar */
    header { height:48px; display:flex; align-items:center; gap:12px;
             padding:0 14px; background:{{PANEL}}; border-bottom:1px solid {{BORD}}; }
    .brand { font-weight:700; letter-spacing:.5px; color:{{ACCENT}}; }
    .pill { display:inline-block; padding:.18rem .6rem; border:1px solid {{BORD}}; border-radius:999px; font-size:12px; color:{{SUBFG}}; }
    .ok { color:#9f9; border-color:#2a4; } .bad { color:#f99; border-color:#933; }
    .spacer { flex:1 1 auto; }

    /* Layout: hero left, tiles right; fixed viewport (no scrolling) */
    .wrap { height: calc(100vh - 48px); display:grid; grid-template-columns: 2fr 1fr; grid-template-rows: 1fr; gap:10px; padding:10px; }
    .hero { background:#000; border:1px solid {{BORD}}; border-radius:10px; overflow:hidden; display:flex; align-items:center; justify-content:center; }
    .hero img { width:100%; height:100%; object-fit:contain; background:#000; }

    .panel { display:grid; grid-template-rows: auto 1fr; gap:8px; }
    .controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    button, .btn { padding:.5rem .8rem; background:{{PANEL}}; color:{{FG}}; border:1px solid {{BORD}}; border-radius:8px; cursor:pointer; }
    button:hover, .btn:hover { border-color:{{ACCENT}}; }

    .tiles { display:grid; grid-template-columns: 1fr 1fr; grid-auto-rows: minmax(0, 1fr); gap:8px; min-height:0; }
    .tile { background:#000; border:1px solid {{BORD}}; border-radius:10px; overflow:hidden; position:relative; display:flex; align-items:center; justify-content:center; }
    .tile img { width:100%; height:100%; object-fit:contain; background:#000; }
    .tile .mini { position:absolute; top:6px; right:6px; display:flex; gap:6px; }
    .mini button { font-size:11px; padding:.25rem .45rem; border-radius:6px; }

    small { color:{{SUBFG}}; }
  </style>
</head>
<body>
  <header>
    <div class="brand">{{theme_name}}</div>
    <span class="pill" id="pwm">PWM: ?</span>
    <span class="pill" id="usb">USB: ?</span>
    <span class="pill" id="csis">CSI: ?</span>
    <div class="spacer"></div>
    <small>http://{{host}}:{{port}}</small>
  </header>

  <div class="wrap">
    <section class="hero">
      <img id="hero" src="/usb/stream/PRIMARY" />
    </section>

    <section class="panel">
      <div class="controls">
        <button onclick="post('/auto/start').then(status)">AUTO</button>
        <button onclick="post('/auto/stop').then(status)">MANUAL</button>
        <label><input type="checkbox" id="ax_x" checked onchange="pushAxes()"> X</label>
        <label><input type="checkbox" id="ax_y" checked onchange="pushAxes()"> Y</label>
        <label><input type="checkbox" id="ax_z" onchange="pushAxes()"> Z</label>
        <a class="btn" href="/health" target="_blank">health</a>
      </div>

      <div class="tiles" id="tiles"></div>
    </section>
  </div>

<script>
async function post(url, data){
  return fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: data ? JSON.stringify(data) : null
  });
}

async function status(){
  try{
    const r = await fetch('/health'); const j = await r.json();
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
  }catch(e){}
}
setInterval(status, 1500); status();

async function pushAxes(){
  const x = document.getElementById('ax_x').checked;
  const y = document.getElementById('ax_y').checked;
  const z = document.getElementById('ax_z').checked;
  await post('/auto/axes', {enable_x:x, enable_y:y, enable_z:z});
}

async function setPrimary(idx){
  await post('/usb/primary/'+idx);
  document.getElementById('hero').src = '/usb/stream/PRIMARY?t=' + Date.now();
  status();
}

async function loadTiles(){
  const tiles = document.getElementById('tiles');
  tiles.innerHTML = '';
  // USB tiles
  try{
    const r = await fetch('/usbs'); const j = await r.json();
    (j.items||[]).forEach(c=>{
      const card = document.createElement('div');
      card.className = 'tile';
      card.innerHTML = `
        <div class="mini">
          <button onclick="setPrimary(${c.idx})">hero</button>
        </div>
        <img src="/usb/stream/${c.idx}" />
      `;
      tiles.appendChild(card);
    });
  }catch(e){}
  // CSI tiles
  try{
    const r = await fetch('/csis'); const j = await r.json();
    (j.items||[]).forEach(c=>{
      const card = document.createElement('div');
      card.className = 'tile';
      card.innerHTML = `
        <div class="mini"></div>
        <a href="/csi/${c.idx}.mjpg" target="_blank">
          <img src="/csi/${c.idx}.mjpg" />
        </a>
      `;
      tiles.appendChild(card);
    });
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
        theme_name=THEME_NAME, host=HOST, port=PORT,
        ACCENT=ACCENT, BG=BG, PANEL=PANEL, BORD=BORD, FG=FG, SUBFG=SUBFG
    )

# USB streams / listing / primary
@app.route("/usb/stream/<sel>")
def usb_stream(sel: str):
    if sel.upper() == "PRIMARY": idx = usbs.primary_idx()
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

@app.route("/usb/primary/<int:idx>", methods=["POST"])
def set_primary(idx: int):
    ok = usbs.set_primary(idx)
    return jsonify({"ok": ok, "primary": usbs.primary_idx()}), (200 if ok else 404)

@app.route("/usbs")
def list_usbs():
    items=[]
    for i, node in usbs.list():
        w = usbs.get(i)
        items.append({"idx": i, "node": node, "ok": bool(w and w.ok), "err": getattr(w, "last_error", "")})
    return jsonify({"items": items, "count": len(items)})

# CSI streams / listing
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

# AUTO / MANUAL (POST for reliability); axis toggles live-update
@app.route("/auto/start", methods=["POST"])
def auto_start():
    control.set_mode("auto")
    return jsonify({"ok": True, "mode": control.mode})

@app.route("/auto/stop", methods=["POST"])
def auto_stop():
    control.set_mode("manual")
    control.set_all_mid()
    return jsonify({"ok": True, "mode": control.mode})

@app.route("/auto/axes", methods=["POST"])
def auto_axes():
    j = request.get_json(force=True, silent=True) or {}
    kw = {}
    for k in ("enable_x","enable_y","enable_z"):
        if k in j: kw[k] = j[k]
    control.set_auto_params(**kw)
    return jsonify({"ok": True, "cfg": control.auto_cfg.__dict__})

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
        "mode": getattr(globals().get("control", None), "mode", "unknown"),
        "motors": [{"ch": m["ch"], "pos": int(m["pos"]), "mid": m["mid"]} for m in MOTORS],
        "targets": getattr(globals().get("control", None), "targets", []),
        "auto_axes": {
            "enable_x": control.auto_cfg.enable_x,
            "enable_y": control.auto_cfg.enable_y,
            "enable_z": control.auto_cfg.enable_z,
        },
        "usb_count": len(usb_items),
        "usb_primary": usbs.primary_idx(),
        "usb": usb_items,
        "csi_count": len(csi_items),
        "csi": csi_items,
        "picamera2": PICAM_AVAILABLE,
        "csi_cfg": {"w": CSI_W, "h": CSI_H, "fps": CSI_FPS, "q": CSI_JPEG_QUALITY}
    })

# ---------------- LIFECYCLE ----------------
def start_threads():
    usbs.start(MAX_USB_CAMS)
    if PICAM_AVAILABLE:
        try:
            desired = [int(x) for x in CSI_INDEXES_ENV] if CSI_INDEXES_ENV else []
        except Exception:
            desired = []
        csis.start(desired)
    else:
        print("[CSI] picamera2 not available; skipping", flush=True)
    threading.Thread(target=pwm_watchdog, daemon=True).start()
    globals()["control"] = MotorController(MOTORS, pwm, pwm_lock)
    control.start()
    control.set_mode("auto")  # ensure AUTO at boot

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
    print("\n[EXIT] cleaned up.", flush=True)
    try: time.sleep(0.3)
    except: pass
    os._exit(0)

def main():
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    start_threads()
    print(f"[READY] http://{HOST}:{PORT} (Hero /usb/stream/PRIMARY, tiles USB+CSI)", flush=True)
    app.run(host=HOST, port=PORT, threaded=True)

if __name__ == "__main__":
    main()
