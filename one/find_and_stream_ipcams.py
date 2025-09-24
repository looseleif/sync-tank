#!/usr/bin/env python3
import os, sys, time, subprocess, re, threading, ipaddress, json
from typing import List, Dict, Optional, Tuple
from flask import Flask, Response, render_template_string
import cv2

print("[BOOT] starting scanner/server...", flush=True)

# Force OpenCV to use TCP for RTSP (usually more reliable)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# --------- CONFIG ---------
PORT = 8002
FRAME_W, FRAME_H, FRAME_FPS = 1280, 720, 15
JPEG_QUALITY = 80

# Defaults (can be overridden via env):
# Comma-separated subnets; we'll merge with auto-inferred subnet.
DEFAULT_EXTRA_CIDRS = ["192.168.0.0/24", "192.168.1.0/24"]
DEFAULT_RTSP_PORTS = [554, 8554]  # many cams use 8554
USERNAME = os.getenv("RTSP_USER", "admin")
PASSWORD = os.getenv("RTSP_PASS", "admin")

# Common RTSP paths to probe
COMMON_RTSP_PATHS = [
    # Hikvision
    "/Streaming/Channels/101", "/Streaming/Channels/102",
    # Dahua / Amcrest
    "/cam/realmonitor?channel=1&subtype=0", "/cam/realmonitor?channel=1&subtype=1",
    # Reolink
    "/h264Preview_01_main", "/h264Preview_01_sub",
    # TP-Link Tapo
    "/stream1", "/stream2",
    # Uniview / generic
    "/LiveMedia/ch1/Media1", "/live", "/main",
]
# --------------------------

# Env overrides
ENV_CIDRS = [c.strip() for c in os.getenv("CIDRS", "").split(",") if c.strip()]
ENV_PORTS = [int(p) for p in os.getenv("PORTS", "").split(",") if p.strip().isdigit()]

HTML = """
<!doctype html>
<html>
  <head>
    <title>Discovered IP Cameras</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
      :root { color-scheme: dark; }
      body { font-family: system-ui, sans-serif; margin: 0; background: #0b0b0b; color: #eee; }
      header { padding: .75rem 1rem; background: #161616; border-bottom: 1px solid #222; position: sticky; top: 0; }
      .grid { display: grid; gap: 12px; padding: 12px; grid-template-columns: 1fr; }
      @media (min-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } }
      @media (min-width: 1400px) { .grid { grid-template-columns: 1fr 1fr 1fr; } }
      .card { background: #000; border: 1px solid #1f1f1f; border-radius: 10px; overflow: hidden; }
      .card h2 { margin: 0; padding: 8px 12px; background: #222; font-size: 14px; border-bottom: 1px solid #1f1f1f; }
      .card .sub { color: #9aa; font-size: 12px; padding: 8px 12px; }
      .card img { width: 100%; display: block; background: #000; aspect-ratio: 16/9; object-fit: contain; }
      .hint { padding: 0 12px 12px; color: #9aa; font-size: 12px; }
      code { background: #161616; padding: 2px 6px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <header><strong>Discovered IP Cameras</strong> — MJPEG — http://{{host}}:{{port}}</header>
    <div class="grid">
      {% for cam in cams %}
      <div class="card">
        <h2>{{cam["name"]}}</h2>
        <div class="sub">{{cam["url"]}}</div>
        <img src="{{ url_for('stream_cam', idx=loop.index0) }}" />
      </div>
      {% endfor %}
    </div>
    <div class="hint">
      Endpoints: /cam/0 ... /cam/{{ cams|length - 1 }} — restart the script to rescan.
    </div>
  </body>
</html>
"""

app = Flask(__name__)

# ---------------- Frame Buffer ----------------
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

# ---------------- Helpers ----------------
def infer_primary_cidr() -> Optional[str]:
    """Return primary IPv4 CIDR (e.g. '10.0.0.0/24') if possible."""
    try:
        out = subprocess.run(["ip", "-j", "-4", "address"], capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        for ifc in data:
            if ifc.get("operstate") != "UP":
                continue
            for a in ifc.get("addr_info", []):
                if a.get("family") == "inet":
                    local, prefix = a.get("local"), a.get("prefixlen")
                    if local and prefix:
                        net = ipaddress.ip_network(f"{local}/{prefix}", strict=False)
                        print(f"[INFO] Inferred CIDR: {net}", flush=True)
                        return str(net)
    except Exception as e:
        print(f"[WARN] CIDR inference failed: {e}", flush=True)
    return None

def nmap_scan_rtsp(cidr: str, port: int) -> List[str]:
    """Use nmap to find hosts with given RTSP port open. Return list of IPs."""
    print(f"[SCAN] nmap {cidr} for RTSP tcp/{port}", flush=True)
    try:
        res = subprocess.run(
            ["nmap", "-p", str(port), "--open", cidr, "-oG", "-"],
            capture_output=True, text=True, check=True
        )
    except Exception as e:
        print(f"[ERROR] nmap failed on {cidr}:{port}: {e}", flush=True)
        return []
    ips = []
    for line in res.stdout.splitlines():
        if "Ports:" in line and f"{port}/open" in line:
            m = re.search(r"Host:\s+(\S+)\s", line)
            if m:
                ips.append(m.group(1))
    return ips

def unique(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def build_targets() -> Tuple[List[str], List[int]]:
    cidrs = []
    ports = ENV_PORTS if ENV_PORTS else DEFAULT_RTSP_PORTS

    # Auto + defaults + env CIDRS
    primary = infer_primary_cidr()
    if primary: cidrs.append(primary)
    cidrs.extend(DEFAULT_EXTRA_CIDRS)
    cidrs.extend(ENV_CIDRS)
    # De-dupe and keep only valid networks
    out_cidrs = []
    for c in cidrs:
        try:
            ipaddress.ip_network(c, strict=False)
            out_cidrs.append(c)
        except Exception:
            print(f"[WARN] Ignoring invalid CIDR: {c}", flush=True)
    out_cidrs = unique(out_cidrs)
    print(f"[INFO] Will scan CIDRs: {out_cidrs}", flush=True)
    print(f"[INFO] Will scan RTSP ports: {ports}", flush=True)
    return out_cidrs, ports

def try_open_rtsp(ip: str, port: int) -> Optional[str]:
    """Try common RTSP paths with provided creds. Return first working URL."""
    for path in COMMON_RTSP_PATHS:
        url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:{port}{path}"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS,          FRAME_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            print(f"[PROBE] OK: {url}", flush=True)
            return url
        else:
            print(f"[PROBE] fail: {url}", flush=True)
    return None

def bgr_to_jpeg(bgr, quality=JPEG_QUALITY) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None

def camera_loop(name: str, url: str, fb: FrameBuffer):
    """Read frames forever; auto-reconnect on stall."""
    period = 1.0 / float(max(1, FRAME_FPS))
    while True:
        cap = None
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS,          FRAME_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)
            time.sleep(0.2)
            last_ok = time.time()
            while True:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if time.time() - last_ok > 3.0:
                        raise RuntimeError("stalled stream")
                    time.sleep(0.05)
                    continue
                last_ok = time.time()
                jpg = bgr_to_jpeg(frame, JPEG_QUALITY)
                if jpg: fb.set(jpg)
                dt = time.time() - t0
                if dt < period: time.sleep(period - dt)
        except Exception as e:
            print(f"[{name}] reconnecting after error: {e}", flush=True)
            time.sleep(1.0)
        finally:
            if cap: cap.release()

# ---------------- Flask ----------------
cams: List[Dict[str, str]] = []
buffers: List[FrameBuffer] = []

@app.route("/")
def index():
    return render_template_string(HTML, host="0.0.0.0", port=PORT, cams=cams)

@app.route("/cam/<int:idx>")
def stream_cam(idx: int):
    if idx < 0 or idx >= len(buffers):
        return "Camera not available", 404
    fb = buffers[idx]
    def gen():
        boundary = b"--frame"
        while True:
            frame = fb.get()
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                   frame + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------------- Main ----------------
def main():
    print("[MAIN] entering main()", flush=True)
    cidrs, ports = build_targets()

    all_hosts = []
    for c in cidrs:
        for p in ports:
            hosts = nmap_scan_rtsp(c, p)
            all_hosts.extend([(h, p) for h in hosts])
    # de-dupe by (ip,port)
    seen = set(); targets = []
    for ip, p in all_hosts:
        key = (ip, p)
        if key not in seen:
            seen.add(key); targets.append((ip, p))
    print(f"[SCAN] Candidates: {targets}", flush=True)

    for ip, p in targets:
        url = try_open_rtsp(ip, p)
        if url:
            name = f"Cam @ {ip}:{p}"
            fb = FrameBuffer()
            th = threading.Thread(target=camera_loop, args=(name, url, fb), daemon=True)
            th.start()
            cams.append({"name": name, "url": url})
            buffers.append(fb)

    if not cams:
        print("[WARN] No cameras opened. They may be on another VLAN/subnet, use different creds/paths, or RTSP is disabled.", flush=True)
        print("      Try setting env vars, e.g.:", flush=True)
        print("      CIDRS=10.0.0.0/24,192.168.1.0/24 PORTS=554,8554 RTSP_USER=admin RTSP_PASS=admin", flush=True)
    else:
        print(f"[INFO] Streaming {len(cams)} cam(s) on http://0.0.0.0:{PORT}/", flush=True)

    app.run(host="0.0.0.0", port=PORT, threaded=True)

if __name__ == "__main__":
    main()
