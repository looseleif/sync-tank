#!/usr/bin/env python3
import os, sys, time, signal, glob
from threading import Thread, Event, Lock
from typing import Optional, Dict, List

from flask import Flask, jsonify, Response, render_template_string
import cv2

# ---- Optional: I2C servo control (PCA9685 @ 0x40). If missing, we no-op gracefully.
USE_PWM = True
try:
    import smbus2
except Exception:
    USE_PWM = False

# ---------------- CONFIG ----------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
FRAME_W = int(os.environ.get("FRAME_W", "1280"))
FRAME_H = int(os.environ.get("FRAME_H", "720"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
CAM_SWAP_SEC = float(os.environ.get("CAM_SWAP_SEC", "4.0"))  # how often to swap cams

# Motors: 3 simple channels with midpoint + nudge window
MOTORS = [
    {"ch": 0, "mid": 400, "nudge": 30, "pos": 400},  # forward/back
    {"ch": 1, "mid": 400, "nudge": 30, "pos": 400},  # left/right
    {"ch": 2, "mid": 400, "nudge": 30, "pos": 400},  # up/down
]
CLAMP = (0, 4095)

# Map UI commands -> (index, +1/-1) or special "stop"
CMD_MAP: Dict[str, Optional[tuple]] = {
    "w": (0, +1), "s": (0, -1),
    "a": (1, +1), "d": (1, -1),
    "q": (2, +1), "e": (2, -1),
    "stop": None,
}

# --------------- CAMERA DISCOVERY -----------------
def discover_by_id_nodes() -> List[str]:
    """
    Return a stable, sorted list of by-id video nodes (index0 only),
    falling back to /dev/videoN if by-id not present.
    """
    nodes = sorted(
        p for p in glob.glob("/dev/v4l/by-id/*-video-index0")
        if os.path.islink(p) or os.path.exists(p)
    )
    if nodes:
        return [os.path.realpath(p) for p in nodes]
    # Fallback: just probe /dev/video0..3
    fallback = [p for p in ["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3"] if os.path.exists(p)]
    return fallback

def open_uvc(node: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # lowers CPU if supported
    if not cap.isOpened():
        try: cap.release()
        except: pass
        return None
    return cap

# --------------- PCA9685 ----------------
class PWM:
    def __init__(self, addr=0x40, bus_id=1):
        self.ok = False
        if not USE_PWM:
            return
        try:
            self.bus = smbus2.SMBus(bus_id)
            self.addr = addr
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
        except Exception as e:
            print(f"[PWM] Disabled (I2C error): {e}", flush=True)
            self.ok = False

    def set_pwm(self, ch: int, on: int, off: int):
        if not self.ok:
            return
        off = max(CLAMP[0], min(CLAMP[1], off))
        base = 0x06 + 4 * ch
        self.bus.write_byte_data(self.addr, base + 0, on & 0xFF)
        self.bus.write_byte_data(self.addr, base + 1, (on >> 8) & 0xFF)
        self.bus.write_byte_data(self.addr, base + 2, off & 0xFF)
        self.bus.write_byte_data(self.addr, base + 3, (off >> 8) & 0xFF)

pwm = PWM()
for m in MOTORS:
    m["pos"] = m["mid"]
    pwm.set_pwm(m["ch"], 0, m["mid"])

# --------------- FLASK APP --------------
app = Flask(__name__)
_stop = Event()

# Camera state
_cam_nodes: List[str] = []
_caps: List[Optional[cv2.VideoCapture]] = []
_names: List[str] = []
_active_idx = 0
_switch_lock = Lock()

PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>REEFLEX — USB ROV (Cycling Cams)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; background:#111; color:#eee; font-family: system-ui, sans-serif; }
    header { padding:.75rem 1rem; background:#161616; border-bottom:1px solid #222; position:sticky; top:0 }
    main { padding:12px; display:grid; gap:12px; max-width:1100px; margin:0 auto; }
    img { width:100%; max-height:70vh; object-fit:contain; background:#000; border:1px solid #222; border-radius:10px }
    .row { display:flex; flex-wrap:wrap; gap:8px }
    button { padding:.6rem .9rem; background:#262626; color:#eee; border:1px solid #333; border-radius:8px; cursor:pointer; }
    button:hover { background:#2e2e2e }
    small { color:#9aa }
  </style>
</head>
<body>
  <header><strong>REEFLEX — USB ROV</strong> — cycling cams every {{swap}}s — http://{{host}}:{{port}}</header>
  <main>
    <img src="/video" />
    <div><small>Active camera: <span id="camname"></span> (<a href="/cams" target="_blank">list</a>)</small></div>
    <div class="row">
      <button onclick="send('w')">Forward (W)</button>
      <button onclick="send('s')">Back (S)</button>
      <button onclick="send('a')">Left (A)</button>
      <button onclick="send('d')">Right (D)</button>
      <button onclick="send('q')">Up (Q)</button>
      <button onclick="send('e')">Down (E)</button>
      <button onclick="send('stop')">Stop</button>
    </div>
    <pre id="status"></pre>
  </main>
<script>
async function send(cmd){
  const r = await fetch('/cmd/'+cmd);
  const t = await r.text();
  document.getElementById('status').textContent = t;
}
document.addEventListener('keydown', (e)=>{
  const k = e.key.toLowerCase();
  if(['w','a','s','d','q','e'].includes(k)) send(k);
  if(k==='x' || k===' ') send('stop');
});
async function pollName(){
  try{
    const r = await fetch('/which'); 
    const j = await r.json();
    document.getElementById('camname').textContent = j.active_name;
  }catch(e){}
  setTimeout(pollName, 1000);
}
pollName();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(PAGE, host=HOST, port=PORT, swap=CAM_SWAP_SEC)

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
    def gen():
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        font = cv2.FONT_HERSHEY_SIMPLEX
        while not _stop.is_set():
            with _switch_lock:
                idx = _active_idx
                cap = _caps[idx] if idx < len(_caps) else None
                label = _names[idx] if idx < len(_names) else "N/A"

            ok, frame = cap.read() if cap is not None else (False, None)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # optionally enforce size
            if FRAME_W and FRAME_H:
                h, w = frame.shape[:2]
                if abs(w-FRAME_W) > 8 or abs(h-FRAME_H) > 8:
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

            # overlay which cam is live
            try:
                cv2.putText(frame, f"{label}", (12, 28), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
            except Exception:
                pass

            ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cmd/<c>")
def cmd(c: str):
    c = c.lower()
    if c not in CMD_MAP and c != "stop":
        return f"ERR unknown cmd {c}", 400

    if c == "stop":
        for m in MOTORS:
            m["pos"] = m["mid"]
            pwm.set_pwm(m["ch"], 0, m["mid"])
        return "OK stop -> midpoints"

    idx, sign = CMD_MAP[c]
    m = MOTORS[idx]
    lower = m["mid"] - m["nudge"] * 5
    upper = m["mid"] + m["nudge"] * 5
    target = m["pos"] + (m["nudge"] * sign)
    m["pos"] = max(lower, min(upper, target))
    pwm.set_pwm(m["ch"], 0, int(m["pos"]))
    return f"OK {c}: ch={m['ch']} pos={int(m['pos'])}"

@app.route("/health")
def health():
    with _switch_lock:
        name = _names[_active_idx] if _names else "<none>"
    return jsonify({
        "motors": [{"ch": m["ch"], "pos": m["pos"], "mid": m["mid"]} for m in MOTORS],
        "active_cam_index": _active_idx,
        "active_cam_name": name,
        "frame_w": FRAME_W, "frame_h": FRAME_H, "fps": FRAME_FPS,
        "swap_sec": CAM_SWAP_SEC
    })

def switcher():
    global _active_idx
    last = time.time()
    while not _stop.is_set():
        time.sleep(0.1)
        if time.time() - last >= CAM_SWAP_SEC and len(_caps) > 1:
            with _switch_lock:
                _active_idx = (_active_idx + 1) % len(_caps)
            last = time.time()

def cleanup_and_exit(*_):
    _stop.set()
    try:
        for m in MOTORS: pwm.set_pwm(m["ch"], 0, m["mid"])
    except: pass
    for c in _caps:
        try:
            if c: c.release()
        except: pass
    print("\n[EXIT] cleaned up.", flush=True)
    sys.exit(0)

def main():
    global _cam_nodes, _caps, _names, _active_idx
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    nodes = discover_by_id_nodes()
    if not nodes:
        raise RuntimeError("No video devices found")

    # Pick up to two cams; if you want all, keep them all—cycling logic handles len>1
    _cam_nodes = nodes
    _names = []
    _caps = []
    for n in _cam_nodes:
        cap = open_uvc(n)
        if cap:
            _caps.append(cap)
            _names.append(os.path.basename(n))
            print(f"[CAM] opened {n}", flush=True)
        else:
            print(f"[CAM] failed to open {n}", flush=True)

    if not _caps:
        raise RuntimeError("No cameras could be opened")

    # Clamp to at least one, we can cycle across however many opened
    _active_idx = 0

    # Start switcher thread
    Thread(target=switcher, daemon=True).start()

    print(f"[READY] open http://{HOST}:{PORT}  (video at /video, cmds at /cmd/<w|a|s|d|q|e|stop>)", flush=True)
    print(f"[INFO] cycling across {len(_caps)} cams every {CAM_SWAP_SEC}s", flush=True)
    app.run(host=HOST, port=PORT, threaded=True)

if __name__ == "__main__":
    main()
