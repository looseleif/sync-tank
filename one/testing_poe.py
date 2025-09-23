# --- TUNING KNOBS ---
TARGET_WIDTH   = 800    # downscale width (keep aspect)
TARGET_FPS     = 10     # encode/send at most this many frames per second
JPEG_QUALITY   = 45     # 1..100 (lower = more compression, smaller/faster)
USE_UDP        = False  # try True if your LAN is super clean/low-loss
# ---------------------

import os, time, threading, cv2
from collections import deque
from flask import Flask, Response, abort, jsonify, render_template_string

# Use FFmpeg backend with RTSP options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    f"rtsp_transport;{'udp' if USE_UDP else 'tcp'}"
    "|stimeout;4000000|max_delay;500000|buffer_size;102400"
)

CAMERAS = {
    "cam108": "rtsp://192.168.1.108:554/user=admin&password=admin&channel=1&stream=1.sdp?real_stream",
    "cam185": "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/102",
}

app = Flask(__name__)

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>SYNC TANK</title>
<style>body{font-family:system-ui;margin:2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}
img{width:100%;border-radius:10px}</style></head><body>
<h1>SYNC TANK</h1><div class="grid">
{% for n in names %}<div><h3>{{n}}</h3><a href="/cam/{{n}}"><img src="/cam/{{n}}"></a></div>{% endfor %}
</div></body></html>"""

class RTSPReader:
    def __init__(self, name, url):
        self.name, self.url = name, url
        self.cap = None
        self.last_jpeg = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._next_deadline = 0.0

    def start(self): self.thread.start()
    def stop(self):  self.running = False

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def _loop(self):
        backoff = 1.0
        while self.running:
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    time.sleep(backoff); backoff = min(10, backoff*2); continue
                backoff = 1.0

            ok, frame = self.cap.read()
            if not ok or frame is None:
                try: self.cap.release()
                except: pass
                self.cap = None
                time.sleep(backoff); backoff = min(10, backoff*2)
                continue

            # Downscale for less CPU/bandwidth
            if TARGET_WIDTH:
                h, w = frame.shape[:2]
                if w > TARGET_WIDTH:
                    new_h = int(h * (TARGET_WIDTH / w))
                    frame = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

            # FPS cap: only encode if we reached the next deadline
            now = time.perf_counter()
            if now < self._next_deadline:
                continue
            period = 1.0 / max(1, TARGET_FPS)
            self._next_deadline = now + period

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with self.lock:
                    self.last_jpeg = jpg.tobytes()

    def mjpeg(self):
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while self.running:
            with self.lock:
                buf = self.last_jpeg
            if buf is not None:
                # send only the latest (no backlog) to keep latency low
                yield boundary + buf + b"\r\n"
            else:
                time.sleep(0.03)

readers = {}

@app.before_first_request
def start_readers():
    for name, url in CAMERAS.items():
        r = RTSPReader(name, url)
        r.start()
        readers[name] = r

@app.route("/")
def index(): return render_template_string(INDEX_HTML, names=list(CAMERAS.keys()))
@app.route("/cameras")
def cameras(): return jsonify({"cameras": list(CAMERAS.keys())})
@app.route("/cam/<name>")
def cam(name):
    r = readers.get(name)
    if not r: return abort(404, f"Unknown camera '{name}'")
    return Response(r.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/health")
def health(): return jsonify({n: (rd.last_jpeg is not None) for n, rd in readers.items()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
