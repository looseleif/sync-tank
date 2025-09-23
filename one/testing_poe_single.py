# single_stream.py
# Run:
#   sudo apt update && sudo apt install -y python3-flask python3-opencv ffmpeg
#   python3 single_stream.py
# Then open: http://<PI-WIFI-IP>:8000/  (the page embeds /video)

import os, time, threading, cv2
from flask import Flask, Response, render_template_string, jsonify

# ====== SELECT ONE CAMERA URL (H.264 substream preferred) ======
# XMeye/Xiongmai-style (your 108 cam): use substream (usually H.264)
CAM_URL = "rtsp://192.168.1.108:554/user=admin&password=admin&channel=1&stream=1.sdp?real_stream"
# Hikvision-style (your 185 cam) H.264 substream example:
# CAM_URL = "rtsp://admin:admin@192.168.1.185:554/Streaming/Channels/102"
# ================================================================

# --- Tuning (lower for smoother playback) ---
TARGET_WIDTH  = 200   # downscale width (keep aspect). Try 360 if needed.
TARGET_FPS    = 2     # server-side cap. Try 6 if still choppy.
JPEG_QUALITY  = 15    # 1..100 (lower=smaller/faster)
USE_UDP       = False # keep TCP unless your LAN is very clean
# -------------------------------------------

# Make OpenCV use FFmpeg with sensible RTSP options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    f"rtsp_transport;{'udp' if USE_UDP else 'tcp'}"
    "|stimeout;5000000|max_delay;500000|buffer_size;102400"
)

app = Flask(__name__)

INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Single Stream</title>
<style>body{font-family:system-ui;margin:2rem} img{max-width:100%;border-radius:10px}</style>
</head><body>
<h1>SYNC TANK — Single Stream</h1>
<p>If it stutters, reduce width/FPS in the script, and ensure your camera substream is H.264 at low res/fps/bitrate.</p>
<img src="/video">
</body></html>
"""

class SingleRTSP:
    """Persistent RTSP -> JPEG bridge with auto-reconnect and throttling."""
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.lock = threading.Lock()
        self.last_jpeg = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.period = 1.0 / max(1, TARGET_FPS)
        self.next_deadline = 0.0

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
                    time.sleep(backoff); backoff = min(backoff * 2, 10); continue
                backoff = 1.0

            ok, frame = self.cap.read()
            if not ok or frame is None:
                try: self.cap.release()
                except: pass
                self.cap = None
                time.sleep(backoff); backoff = min(backoff * 2, 10)
                continue

            # Downscale to lighten CPU/bandwidth
            if TARGET_WIDTH:
                h, w = frame.shape[:2]
                if w > TARGET_WIDTH:
                    new_h = int(h * (TARGET_WIDTH / w))
                    frame = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

            # FPS cap: only encode at most TARGET_FPS
            now = time.perf_counter()
            if now < self.next_deadline:
                continue
            self.next_deadline = now + self.period

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
                yield boundary + buf + b"\r\n"
            else:
                time.sleep(0.03)

streamer = SingleRTSP(CAM_URL)
streamer.start()

@app.route("/")
def index(): return render_template_string(INDEX_HTML)

@app.route("/video")
def video():
    return Response(streamer.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    return jsonify({"has_frame": streamer.last_jpeg is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
