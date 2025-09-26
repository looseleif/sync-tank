#!/usr/bin/env python3
# rtsp_test.py — ultra-lean RTSP snapshot/MJPEG tester

import os, time, threading, subprocess
from typing import Optional, Dict, List, Tuple
from flask import Flask, jsonify, Response, make_response, render_template_string, abort
import cv2
import numpy as np

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8081"))

# Snapshot cadence (seconds). Keep light.
RTSP_SNAPSHOT_SEC = float(os.environ.get("RTSP_SNAPSHOT_SEC", "5.0"))
RTSP_MAX_PAR = int(os.environ.get("RTSP_MAX_PAR", "1"))
KNOWN_URLS = [
    # replace or extend as needed
    "rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream",
    "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/101",
    "rtsp://admin:admin@192.168.0.108:554/h264/ch1/main/av_stream",
]

# Nudge FFmpeg/RTSP behavior
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|rw_timeout;1200000|stimeout;1200000")

app = Flask(__name__)
_stop = threading.Event()
rtsp_sem = threading.Semaphore(max(1, RTSP_MAX_PAR))

class FrameBuffer:
    def __init__(self, w=640, h=360, q=70):
        self._buf: Optional[bytes] = None
        self._cond = threading.Condition()
        img = np.zeros((h, w, 3), dtype=np.uint8)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        self._buf = jpg.tobytes() if ok else None

    def set(self, b: bytes):
        with self._cond:
            self._buf = b
            self._cond.notify_all()

    def peek(self) -> Optional[bytes]:
        return self._buf

    def get(self) -> bytes:
        with self._cond:
            while self._buf is None:
                self._cond.wait()
            return self._buf

class RTSPWorker(threading.Thread):
    def __init__(self, idx:int, url:str):
        super().__init__(daemon=True)
        self.idx=idx; self.url=url
        self.buf=FrameBuffer()
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
            # Sleep to next snapshot cadence
            sleep=max(0.1, RTSP_SNAPSHOT_SEC-(time.time()-t0))
            for _ in range(int(sleep*10)):
                if self._stop.is_set(): break
                time.sleep(0.1)
        print(f"[RTSP{self.idx}] stopped", flush=True)

class RTSPRegistry:
    def __init__(self):
        self._lock=threading.Lock()
        self._cams:Dict[int,RTSPWorker]={}
    def start_all(self, urls:List[str]):
        with self._lock:
            for i,url in enumerate(urls):
                w=RTSPWorker(i,url)
                self._cams[i]=w
                w.start()
    def list(self)->List[Tuple[int,str]]:
        with self._lock:
            return [(i,c.url) for i,c in sorted(self._cams.items())]
    def get(self,idx:int)->Optional[RTSPWorker]:
        with self._lock: return self._cams.get(idx)
    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

rtsp = RTSPRegistry()

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>rtsp_test</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#0b0b0b;color:#eee;font-family:system-ui,sans-serif}
header{padding:.6rem 1rem;background:#141414;border-bottom:1px solid #222}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;padding:8px}
.card{background:#000;border:1px solid #222;border-radius:8px;overflow:hidden}
.card img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;display:block}
small{color:#9aa}
</style></head>
<body>
<header><strong>RTSP Tester</strong> <small>MJPEG: /rtsp/&lt;idx&gt;.mjpg — JPG: /rtsp/&lt;idx&gt;.jpg</small></header>
<div class="grid" id="g"></div>
<script>
async function load(){
  const r=await fetch('/cams'); const j=await r.json();
  const g=document.getElementById('g'); g.innerHTML='';
  (j.items||[]).forEach(c=>{
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<img id="m_${c.idx}" src="/rtsp/${c.idx}.mjpg" />`;
    g.appendChild(d);
  });
}
load();
</script>
</body></html>"""

@app.route("/")
def index():
    return PAGE

@app.route("/cams")
def cams():
    return jsonify({"items":[{"idx":i,"url":u} for i,u in rtsp.list()]})

@app.route("/rtsp/<int:idx>.jpg")
def rtsp_jpg(idx:int):
    w=rtsp.get(idx)
    if not w: abort(404)
    b=w.buf.peek()
    if not b: abort(503,"no frame yet")
    r=make_response(b); r.headers["Content-Type"]="image/jpeg"; return r

@app.route("/rtsp/<int:idx>.mjpg")
def rtsp_mjpg(idx:int):
    w=rtsp.get(idx)
    if not w: abort(404)
    def gen():
        boundary=b"--frame\r\n"
        while not _stop.is_set():
            frame=w.buf.get()
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            # Lightly tick so we don't hog CPU; image updates when worker refreshes
            time.sleep(0.5)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def cleanup(*_):
    _stop.set()
    try: rtsp.stop_all()
    except: pass
    print("\n[EXIT] RTSP tester cleaned up.", flush=True)

def main():
    # start workers
    rtsp.start_all(KNOWN_URLS)
    try:
        print(f"[READY] http://{HOST}:{PORT} (RTSP tester)", flush=True)
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False, debug=False)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl-C", flush=True)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
