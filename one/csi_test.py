#!/usr/bin/env python3
# csi_test.py — ultra-lean CSI Picamera2 MJPEG/JPG tester

import os, time, io, threading
from typing import Optional, Dict, List, Tuple
from flask import Flask, jsonify, Response, make_response, render_template_string, abort

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8082"))
CSI_W = int(os.environ.get("CSI_W", "1280"))
CSI_H = int(os.environ.get("CSI_H", "720"))
CSI_FPS = int(os.environ.get("CSI_FPS", "10"))
CSI_JPEG_QUALITY = int(os.environ.get("CSI_JPEG_QUALITY", "70"))

# Try to import picamera2
try:
    from picamera2 import Picamera2
    from PIL import Image
    PICAM_AVAILABLE = True
except Exception as e:
    PICAM_AVAILABLE = False
    _IMPORT_ERR = str(e)

app = Flask(__name__)
_stop = threading.Event()

class CSICam:
    def __init__(self, idx:int, size=(1280,720), fps:int=10, q:int=70):
        self.idx=idx; self.size=size; self.fps=max(1,min(fps,30)); self.q=max(1,min(q,95))
        self.picam: Optional[Picamera2]=None
        self.latest: Optional[bytes]=None
        self._t: Optional[threading.Thread]=None
        self._stop=threading.Event()
        self.err=""; self.ok=False

    def start(self):
        if not PICAM_AVAILABLE:
            self.err="picamera2 not available"; return
        try:
            self.picam=Picamera2(camera_num=self.idx)
            cfg=self.picam.create_preview_configuration(main={"size":self.size,"format":"RGB888"})
            self.picam.configure(cfg); self.picam.start(); time.sleep(0.3)
            self._stop.clear()
            self._t=threading.Thread(target=self._loop,daemon=True); self._t.start()
            self.ok=True
            print(f"[CSI{self.idx}] started", flush=True)
        except Exception as e:
            self.err=f"start: {e}"
            try:
                if self.picam: self.picam.close()
            except: pass
            self.picam=None

    def stop(self):
        self._stop.set()
        if self._t: self._t.join(timeout=1.0)
        if self.picam:
            try: self.picam.stop()
            except: pass
            try: self.picam.close()
            except: pass
        self.ok=False; self.picam=None

    def _loop(self):
        period=1.0/self.fps
        from PIL import Image
        import io
        while not self._stop.is_set():
            t0=time.time()
            try:
                arr=self.picam.capture_array("main")
                with io.BytesIO() as b:
                    Image.fromarray(arr).save(b, format="JPEG", quality=self.q, optimize=True)
                    self.latest=b.getvalue()
            except Exception as e:
                self.err=f"cap: {e}"
            dt=time.time()-t0
            if dt<period: time.sleep(period-dt)

    def frames(self):
        boundary=b"--frame\r\n"
        while not _stop.is_set():
            if self.latest is None:
                time.sleep(0.05); continue
            yield boundary+b"Content-Type: image/jpeg\r\n\r\n"+self.latest+b"\r\n"
            time.sleep(0.2)

class CSIReg:
    def __init__(self):
        self._lock=threading.Lock()
        self._cams:Dict[int,CSICam]={}
        self._info:List[str]=[]
    def detect(self)->List[int]:
        if not PICAM_AVAILABLE: self._info=[]; return []
        try:
            info=Picamera2.global_camera_info()
            self._info=[str(x) for x in info]
            return list(range(len(info)))
        except Exception:
            self._info=[]; return []
    def start_all(self):
        with self._lock:
            avail=self.detect()
            for i in avail:
                c=CSICam(i,size=(CSI_W,CSI_H),fps=CSI_FPS,q=CSI_JPEG_QUALITY)
                self._cams[i]=c; c.start()
            if not avail: print("[CSI] none detected", flush=True)
    def list(self)->List[int]:
        with self._lock: return sorted(self._cams.keys())
    def get(self, idx:int)->Optional[CSICam]:
        with self._lock: return self._cams.get(idx)
    def stop_all(self):
        with self._lock:
            for c in self._cams.values(): c.stop()

csis = CSIReg()

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>csi_test</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#0b0b0b;color:#eee;font-family:system-ui,sans-serif}
header{padding:.6rem 1rem;background:#141414;border-bottom:1px solid #222}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;padding:8px}
.card{background:#000;border:1px solid #222;border-radius:8px;overflow:hidden}
.card img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;display:block}
small{color:#9aa}
</style></head>
<body>
<header><strong>CSI Tester</strong> <small>MJPEG: /csi/&lt;idx&gt;.mjpg — JPG: /csi/&lt;idx&gt;.jpg</small></header>
<div class="grid" id="g"></div>
<script>
async function load(){
  const r=await fetch('/csis'); const j=await r.json();
  const g=document.getElementById('g'); g.innerHTML='';
  (j.items||[]).forEach(c=>{
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<img src="/csi/${c.idx}.mjpg" />`;
    g.appendChild(d);
  });
}
load();
</script>
</body></html>"""

@app.route("/")
def index():
    if not PICAM_AVAILABLE:
        return f"<pre>picamera2 not available: {_IMPORT_ERR}</pre>", 500
    return PAGE

@app.route("/csis")
def list_csis():
    return jsonify({"items":[{"idx":i} for i in csis.list()]})

@app.route("/csi/<int:idx>.mjpg")
def csi_mjpg(idx:int):
    cam=csis.get(idx)
    if not cam: abort(404)
    return Response(cam.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/csi/<int:idx>.jpg")
def csi_jpg(idx:int):
    cam=csis.get(idx)
    if not cam or cam.latest is None: abort(503,"no frame")
    r=make_response(cam.latest); r.headers["Content-Type"]="image/jpeg"; return r

def cleanup(*_):
    _stop.set()
    try: csis.stop_all()
    except: pass
    print("\n[EXIT] CSI tester cleaned up.", flush=True)

def main():
    if PICAM_AVAILABLE:
        csis.start_all()
    try:
        print(f"[READY] http://{HOST}:{PORT} (CSI tester)", flush=True)
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False, debug=False)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl-C", flush=True)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
