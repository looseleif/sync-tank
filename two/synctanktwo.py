#!/usr/bin/env python3
# synctanktwo — REEFLEX (status-only UI, gamepad+PWM control, USB cam cycling)
# - Focus on /dev/video0 (lowest node) while gamepad is active; otherwise cycle across cams
# - MJPEG stream at /video; same active feed mirrored fullscreen on local Pi display
# - Gamepad (F310/XInput) maps sticks -> 3 PWM channels for the robot arm
# - Clean shutdown via SIGINT/SIGTERM (no Flask reloader)

import os, sys, time, signal, glob, re
from threading import Thread, Event, Lock
from typing import Optional, Dict, List, Tuple

from flask import Flask, jsonify, Response, render_template_string
import cv2
import numpy as np

# ---- Optional: I2C servo control (PCA9685 @ 0x40). If missing, we no-op gracefully.
USE_PWM = True
try:
    import smbus2
except Exception:
    USE_PWM = False

# ---- Evdev for gamepad
USE_GAMEPAD = True
try:
    from evdev import InputDevice, ecodes, list_devices, AbsInfo
except Exception:
    USE_GAMEPAD = False

# ---------------- CONFIG ----------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
CAM_SWAP_SEC = float(os.environ.get("CAM_SWAP_SEC", "4.0"))  # swap cadence when idle

# How long after last gamepad action we keep locking to /dev/video0
GP_FOCUS_TIMEOUT = float(os.environ.get("GP_FOCUS_TIMEOUT", "2.0"))

# Local fullscreen mirror (set 0 to disable)
MIRROR_LOCAL = os.environ.get("MIRROR_LOCAL", "1").lower() not in ("0","false","no")

# Theme (synctankthree vibe)
THEME_TITLE = "synctanktwo"
THEME_NAME  = "REEFLEX"
ACCENT = "#6EE7FF"
BG     = "#07090b"
PANEL  = "#0e1216"
BORD   = "#1a2026"
FG     = "#e8f0f6"
SUBFG  = "#93a3b4"

# Motors: 3 channels with midpoint + span
MOTORS = [
    {"name":"fwd_back", "ch": 0, "mid": 400, "span": 150, "pos": 400},  # forward/back
    {"name":"left_right","ch": 4, "mid": 400, "span": 150, "pos": 400},  # left/right
    {"name":"up_down",  "ch": 8, "mid": 400, "span": 150, "pos": 400},  # up/down
]
CLAMP = (0, 4095)

# Gamepad tuning
GP_DEADZONE = float(os.environ.get("GP_DEADZONE", "0.12"))
GP_SMOOTH   = float(os.environ.get("GP_SMOOTH",   "0.25"))  # EMA smoothing
GP_ENABLED  = True  # toggle with START

print("[BOOT] synctanktwo (REEFLEX) starting...", flush=True)

# --------------- CAMERA DISCOVERY -----------------
def discover_by_id_nodes() -> List[str]:
    nodes = sorted(
        p for p in glob.glob("/dev/v4l/by-id/*-video-index0")
        if os.path.islink(p) or os.path.exists(p)
    )
    if nodes:
        return [os.path.realpath(p) for p in nodes]
    # fallback
    return [p for p in sorted(glob.glob("/dev/video[0-9]*")) if os.path.exists(p)]

def node_num(node_path: str) -> int:
    # Returns numeric N for /dev/videoN if possible; else large sentinel
    m = re.search(r"/dev/video(\d+)", node_path)
    return int(m.group(1)) if m else 9999

# --------------- PCA9685 ----------------
class PWM:
    def __init__(self, addr=0x40, bus_id=1):
        self.ok = False
        self.addr = addr
        self.bus = None
        if not USE_PWM:
            print("[PWM] Disabled (smbus2 not available).", flush=True)
            return
        try:
            self.bus = smbus2.SMBus(bus_id)
            MODE1, PRESCALE = 0x00, 0xFE
            self.bus.write_byte_data(addr, MODE1, 0x00)
            time.sleep(0.005)
            prescale_val = int(25000000.0 / (4096 * 50) - 1)
            self.bus.write_byte_data(addr, MODE1, 0x10)
            self.bus.write_byte_data(addr, PRESCALE, prescale_val)
            self.bus.write_byte_data(addr, MODE1, 0x00)
            time.sleep(0.005)
            self.bus.write_byte_data(addr, MODE1, 0xA1)  # auto-increment + all-call
            self.ok = True
            print("[PWM] OK", flush=True)
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
            self.ok = False
            print(f"[PWM] write fail: {e}", flush=True)

pwm = PWM()
for m in MOTORS:
    m["pos"] = m["mid"]
    pwm.set_pwm(m["ch"], 0, m["mid"])

# --------------- GAMEPAD READER ----------------
_last_gp_activity = 0.0  # epoch seconds; updated on any stick/button action
_last_axes = {"ABS_X":0.0, "ABS_Y":0.0, "ABS_RY":0.0}

class GamepadReader(Thread):
    """
    Logitech F310 (XInput):
      Left stick:  ABS_X (L/R), ABS_Y (F/B)
      Right stick: ABS_RY (U/D)
    Buttons:
      BTN_SOUTH (A) -> center all
      BTN_START     -> toggle GP_ENABLED
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_ev = Event()
        self.dev: Optional[InputDevice] = None

        self.axes = {"ABS_X": 0.0, "ABS_Y": 0.0, "ABS_RY": 0.0}
        self.filtered = {k: 0.0 for k in self.axes}

        self.deadzone = GP_DEADZONE
        self.alpha = GP_SMOOTH
        self.max_step = int(os.environ.get("GP_MAX_SLEW", "8"))
        self.tick_hz = 60.0
        self._absinfo_cache: Dict[str, AbsInfo] = {}

    def find_device(self) -> Optional[InputDevice]:
        candidates = []
        for path in list_devices():
            try:
                d = InputDevice(path)
                name = (d.name or "").lower()
                if any(s in name for s in ("f310","xbox 360","xbox controller","logitech")):
                    candidates.append(d)
            except Exception:
                continue
        for d in candidates:
            caps = d.capabilities(verbose=True)
            if ecodes.EV_ABS in caps:
                return d
        return candidates[0] if candidates else None

    def _absinfo(self, code_name: str) -> Optional[AbsInfo]:
        try:
            code = getattr(ecodes, code_name)
            if code_name not in self._absinfo_cache:
                self._absinfo_cache[code_name] = self.dev.absinfo(code)
            return self._absinfo_cache[code_name]
        except Exception:
            return None

    def _normalize(self, value: int, ai: Optional[AbsInfo]) -> float:
        if not ai or ai.max == ai.min:
            return 0.0
        x = (value - ai.min) / float(ai.max - ai.min) * 2.0 - 1.0
        if abs(x) < self.deadzone:
            x = 0.0
        return max(-1.0, min(1.0, x))

    def _apply_targets(self):
        global _last_gp_activity, _last_axes
        # EMA smoothing
        for k in self.axes:
            self.filtered[k] = (1 - self.alpha) * self.filtered[k] + self.alpha * self.axes[k]

        # Map sticks:
        v_fb = -self.filtered["ABS_Y"]   # left stick vertical: up (−) -> forward (+)
        v_lr =  self.filtered["ABS_X"]   # left stick horizontal
        v_ud = -self.filtered["ABS_RY"]  # right stick vertical: up (−) -> up (+)

        vec = [v_fb, v_lr, v_ud]
        # Detect activity (change beyond a small epsilon or any nonzero after deadzone)
        changed = False
        for key, val in zip(("ABS_Y","ABS_X","ABS_RY"), (v_fb, v_lr, v_ud)):
            if abs(val - _last_axes.get(key.replace("ABS_","ABS_"), 0.0)) > 0.02 or abs(val) > 0.01:
                changed = True
                _last_axes[key] = val
        if changed:
            _last_gp_activity = time.time()

        for i, v in enumerate(vec):
            m = MOTORS[i]
            target = int(m["mid"] + m["span"] * max(-1.0, min(1.0, v)))
            delta = target - m["pos"]
            if delta > self.max_step:
                delta = self.max_step
            elif delta < -self.max_step:
                delta = -self.max_step
            m["pos"] += delta
            pwm.set_pwm(m["ch"], 0, int(m["pos"]))

    def center_all(self):
        global _last_gp_activity
        for m in MOTORS:
            m["pos"] = m["mid"]
            pwm.set_pwm(m["ch"], 0, m["mid"])
        _last_gp_activity = time.time()

    def run(self):
        global GP_ENABLED, _last_gp_activity
        if not USE_GAMEPAD:
            print("[GP] python-evdev not installed; skipping gamepad thread.", flush=True)
            return

        self.dev = self.find_device()
        if not self.dev:
            print("[GP] No gamepad found. Set F310 to XInput (rear switch 'X').", flush=True)
            return

        print(f"[GP] Using device: {self.dev.path} ({self.dev.name})", flush=True)

        try:
            self.dev.set_nonblocking(True)
        except Exception:
            pass

        tick = 1.0 / self.tick_hz
        try:
            while not self.stop_ev.is_set():
                # Drain events
                try:
                    for ev in self.dev.read():
                        if ev.type == ecodes.EV_ABS:
                            code = ecodes.ABS.get(ev.code, ev.code)
                            if code in ("ABS_X","ABS_Y","ABS_RY"):
                                ai = self._absinfo(code)
                                self.axes[code] = self._normalize(ev.value, ai)
                        elif ev.type == ecodes.EV_KEY and ev.value == 1:
                            key = ecodes.BTN.get(ev.code, ev.code)
                            if key == "BTN_SOUTH":
                                self.center_all()
                                print("[GP] STOP/CENTER (A)", flush=True)
                            elif key == "BTN_START":
                                GP_ENABLED = not GP_ENABLED
                                _last_gp_activity = time.time()
                                print(f"[GP] Enabled = {GP_ENABLED}", flush=True)
                except BlockingIOError:
                    pass

                if GP_ENABLED:
                    self._apply_targets()

                time.sleep(tick)
        except OSError:
            print("[GP] Device disconnected.", flush=True)

    def stop(self):
        self.stop_ev.set()

# --------------- FLASK APP + CAPTURE LOOP --------------
app = Flask(__name__)
_stop = Event()

# Camera state
_cam_nodes: List[str] = []
_caps: List[Optional[cv2.VideoCapture]] = []
_names: List[str] = []
_active_idx = 0
_lowest_idx = 0
_switch_lock = Lock()

# Shared latest frame from the *active* camera (for both MJPEG and local mirror)
_latest_jpg: Optional[bytes] = None
_latest_bgr: Optional[np.ndarray] = None
_frame_lock = Lock()

def open_uvc(node: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    if not cap.isOpened():
        try: cap.release()
        except: pass
        return None
    return cap

def choose_lowest_index() -> int:
    # Among opened caps, choose the one whose node has lowest /dev/videoN
    if not _cam_nodes:
        return 0
    pairs = [(i, node_num(p)) for i, p in enumerate(_cam_nodes)]
    pairs.sort(key=lambda x: x[1])
    return pairs[0][0]

def capture_loop():
    """Continuously capture from the *current active* camera and publish shared frames."""
    global _latest_jpg, _latest_bgr
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    while not _stop.is_set():
        with _switch_lock:
            idx = _active_idx
            cap = _caps[idx] if 0 <= idx < len(_caps) else None
            label = _names[idx] if 0 <= idx < len(_names) else "N/A"

        if cap is None:
            time.sleep(0.05)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        # Resize if needed
        if FRAME_W and FRAME_H:
            h, w = frame.shape[:2]
            if abs(w-FRAME_W) > 8 or abs(h-FRAME_H) > 8:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

        # Annotate
        try:
            cv2.putText(frame, f"{label}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        except Exception:
            pass

        # Publish
        ok2, jpg = cv2.imencode(".jpg", frame, enc)
        if ok2:
            with _frame_lock:
                _latest_bgr = frame
                _latest_jpg = jpg.tobytes()

def mjpeg_gen():
    boundary = b"--frame\r\n"
    while not _stop.is_set():
        with _frame_lock:
            buf = _latest_jpg
        if buf is None:
            time.sleep(0.01)
            continue
        yield boundary + b"Content-Type: image/jpeg\r\n" + b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n" + buf + b"\r\n"

def local_mirror_loop():
    """Show the same active feed fullscreen on the local Pi display."""
    if not MIRROR_LOCAL:
        return
    try:
        win = "REEFLEX — Live"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Fullscreen if possible
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        while not _stop.is_set():
            with _frame_lock:
                frame = None if _latest_bgr is None else _latest_bgr.copy()
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit local mirror (server continues)
                    break
            except Exception as e:
                print(f"[MIRROR] display error: {e}", flush=True)
                break
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass
    except Exception as e:
        print(f"[MIRROR] disabled (no GUI?): {e}", flush=True)

def switcher():
    """Pick active camera: lock to lowest (/dev/video0) while gamepad is active; else cycle."""
    global _active_idx
    last_cycle = time.time()
    while not _stop.is_set():
        time.sleep(0.05)
        now = time.time()
        # If recently active gamepad, force focus to lowest index (robot arm cam)
        if (now - _last_gp_activity) <= GP_FOCUS_TIMEOUT:
            with _switch_lock:
                if _active_idx != _lowest_idx:
                    _active_idx = _lowest_idx
            continue
        # Otherwise cycle periodically
        if len(_caps) > 1 and (now - last_cycle) >= CAM_SWAP_SEC:
            with _switch_lock:
                _active_idx = (_active_idx + 1) % len(_caps)
            last_cycle = now

# ---------------- HTML (status-only) ----------------
PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{title}}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    .wm {
    position: fixed;
    right: 10px;
    bottom: 8px;
    color: {{SUBFG}};
    opacity: 0.25;
    font-size: 12px;
    user-select: none;
    pointer-events: none;
    }
    :root { color-scheme: dark; }
    html, body { height:100%; }
    body { margin:0; background:{{BG}}; color:{{FG}}; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; overflow:hidden; }
    header { height:46px; display:flex; align-items:center; gap:12px; padding:0 12px; background:{{PANEL}}; border-bottom:1px solid {{BORD}}; position:sticky; top:0; }
    .brand { font-weight:700; letter-spacing:.4px; color:{{ACCENT}}; }
    .stat  { font-size:12px; color:{{SUBFG}}; }
    main { height: calc(100vh - 46px); padding:8px; }
    .hero { height:100%; border:1px solid {{BORD}}; border-radius:10px; background:#000; display:flex; align-items:center; justify-content:center; }
    .hero img { width:100%; height:100%; object-fit:contain; }
  </style>
</head>
<body>
  <header>
    <div class="brand">{{theme}}</div>
    <div class="stat" id="s_pwm">PWM: …</div>
    <div class="stat" id="s_gp">Gamepad: …</div>
    <div class="stat" id="s_cam">Cam: …</div>
    <div class="stat" id="s_swap">Swap: every {{swap}}s</div>
  </header>
  <main>
    <div class="hero"><img id="hero" src="/video" /></div>
  </main>
<script>
async function refreshStatus(){
  try{
    const r = await fetch('/health'); const j = await r.json();
    document.getElementById('s_pwm').textContent  = 'PWM: ' + (j.pwm_ok ? 'OK' : 'ERR');
    document.getElementById('s_gp').textContent   = 'Gamepad: ' + (j.gp_enabled ? 'ON' : 'OFF');
    const nm = (j.active_name || 'N/A') + (j.locked_to_lowest ? ' (robot)' : '');
    document.getElementById('s_cam').textContent  = 'Cam: ' + nm;
  }catch(e){}
}
setInterval(refreshStatus, 2000); refreshStatus();
</script>
<div class="wm">leifberrytwo</div>
</body>
</html>"""

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template_string(
        PAGE,
        title=THEME_TITLE, theme=THEME_NAME, swap=CAM_SWAP_SEC,
        BG=BG, PANEL=PANEL, BORD=BORD, FG=FG, SUBFG=SUBFG, ACCENT=ACCENT,
        watermark="leifberrytwo",
    )

@app.route("/cams")
def cams():
    return jsonify({
        "nodes": _cam_nodes,
        "names": _names,
        "open": [bool(c and c.isOpened()) for c in _caps],
    })

@app.route("/which")
def which():
    with _switch_lock:
        name = _names[_active_idx] if _names else "<none>"
    return jsonify({"active_index": _active_idx, "active_name": name})

@app.route("/video")
def video():
    return Response(mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    with _switch_lock:
        active_name = _names[_active_idx] if _names else "<none>"
        locked = (_active_idx == _lowest_idx) and ((time.time() - _last_gp_activity) <= GP_FOCUS_TIMEOUT)
    return jsonify({
        "pwm_ok": bool(pwm.ok),
        "gp_enabled": bool(GP_ENABLED),
        "motors": [{"name": m["name"], "ch": m["ch"], "pos": int(m["pos"]), "mid": m["mid"], "span": m["span"]} for m in MOTORS],
        "active_cam_index": _active_idx,
        "active_cam_name": active_name,
        "active_name": active_name,
        "swap_sec": CAM_SWAP_SEC,
        "locked_to_lowest": locked
    })

# ---------------- LIFECYCLE ----------------
def cleanup_and_exit(*_):
    _stop.set()
    try:
        for m in MOTORS:
            pwm.set_pwm(m["ch"], 0, m["mid"])
    except: pass
    for c in _caps:
        try:
            if c: c.release()
        except: pass
    try:
        if gp_thread: gp_thread.stop()
    except: pass
    print("\n[EXIT] cleaned up.", flush=True)
    try: time.sleep(0.25)
    except: pass
    os._exit(0)

def main():
    global _cam_nodes, _caps, _names, _active_idx, _lowest_idx, gp_thread

    signal.signal(signal.SIGINT,  cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    # Cameras
    nodes = discover_by_id_nodes()
    if not nodes:
        raise RuntimeError("No video devices found")

    # Sort by /dev/videoN numeric so "lowest" means index 0 of this list
    nodes = sorted(nodes, key=node_num)

    _cam_nodes = nodes
    _names = [os.path.basename(n) for n in nodes]
    _caps = []
    for n in _cam_nodes:
        cap = open_uvc(n)
        if cap:
            _caps.append(cap)
            print(f"[CAM] opened {n}", flush=True)
        else:
            _caps.append(None)
            print(f"[CAM] failed to open {n}", flush=True)

    # Remove any None caps with their names/nodes (keep aligned)
    live_nodes, live_names, live_caps = [], [], []
    for n, nm, cp in zip(_cam_nodes, _names, _caps):
        if cp is not None:
            live_nodes.append(n); live_names.append(nm); live_caps.append(cp)
    _cam_nodes, _names, _caps = live_nodes, live_names, live_caps

    if not _caps:
        raise RuntimeError("No cameras could be opened")

    _lowest_idx = choose_lowest_index()
    _active_idx = _lowest_idx  # start on robot cam

    # Threads
    Thread(target=capture_loop, daemon=True).start()
    Thread(target=switcher, daemon=True).start()
    if MIRROR_LOCAL:
        Thread(target=local_mirror_loop, daemon=True).start()

    # Start gamepad reader
    gp_thread = None
    if USE_GAMEPAD:
        gp_thread = GamepadReader()
        gp_thread.start()
    else:
        print("[GP] Gamepad disabled (python-evdev missing).", flush=True)

    # Ready
    print(f"[READY] http://{HOST}:{PORT}  (video at /video, status at /health)", flush=True)
    try:
        # No reloader -> predictable Ctrl-C
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False, debug=False)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl-C received", flush=True)
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    main()
