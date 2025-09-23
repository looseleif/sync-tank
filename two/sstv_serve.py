import cv2
import time
import threading
from flask import Flask, Response, render_template_string, redirect, url_for
import os

# === USER CONFIG ===
SHOW_ON_MONITOR = False

# Force-load Raspberry Pi CSI camera driver if present
os.system('sudo modprobe bcm2835-v4l2')

app = Flask(__name__)
lock = threading.Lock()

def detect_cameras():
    working_cams = []
    for idx in range(10):  # test indices 0–9
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ /dev/video{idx} opened and delivered frame.")
                working_cams.append((idx, cap))
            else:
                print(f"⚠️ /dev/video{idx} opened but no valid frame. Skipping.")
                cap.release()
        else:
            print(f"⚠️ Could not open /dev/video{idx}.")
    return working_cams

cameras = detect_cameras()
if not cameras:
    raise RuntimeError("❌ No working cameras found!")

current_camera_idx = 0

PAGE_TEMPLATE = """
<html>
<head>
    <title>Multi-Cam Streamer</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #111; color: #fff; }
        button { margin: 5px; padding: 10px 20px; font-size: 16px; }
        img { border: 2px solid #555; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Multi-Cam Streamer</h1>
    <img src="{{ url_for('video_feed', cam_idx=camera_idx) }}" width="640" height="480">
    <p>Currently showing: <b>{{ camera_name }}</b></p>
    <p>
        {% for idx, name in camera_list %}
            <a href="{{ url_for('switch_camera', cam_idx=idx) }}">
                <button>Switch to {{ name }}</button>
            </a>
        {% endfor %}
    </p>
</body>
</html>
"""

def gen_frames(idx):
    while True:
        with lock:
            ret, frame = cameras[idx][1].read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    camera_list = [(i, f"/dev/video{cam[0]}") for i, cam in enumerate(cameras)]
    return render_template_string(PAGE_TEMPLATE,
                                  camera_idx=current_camera_idx,
                                  camera_name=f"/dev/video{cameras[current_camera_idx][0]}",
                                  camera_list=camera_list)

@app.route('/video_feed/<int:cam_idx>')
def video_feed(cam_idx):
    if 0 <= cam_idx < len(cameras):
        return Response(gen_frames(cam_idx),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid camera index", 404

@app.route('/switch/<int:cam_idx>')
def switch_camera(cam_idx):
    global current_camera_idx
    if 0 <= cam_idx < len(cameras):
        with lock:
            current_camera_idx = cam_idx
        print(f"✅ Switched to /dev/video{cameras[cam_idx][0]} (via web)")
    return redirect(url_for('index'))

def flask_thread():
    app.run(host='0.0.0.0', port=5000, threaded=True)

threading.Thread(target=flask_thread, daemon=True).start()

print("\n✅ Multi-Cam streamer running!")
print("🔗 Open web UI at: http://<raspi-ip>:5000/")
if SHOW_ON_MONITOR:
    print("🖥️ Local monitor showing fullscreen feed.")
else:
    print("🖥️ Local monitor display is DISABLED.")
print("❌ Press CTRL+C to quit.\n")

try:
    if SHOW_ON_MONITOR:
        window_name = 'Local Camera Feed'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            with lock:
                ret, frame = cameras[current_camera_idx][1].read()
            if ret:
                cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
    else:
        while True:
            time.sleep(1)
finally:
    for _, cam in cameras:
        cam.release()
    if SHOW_ON_MONITOR:
        cv2.destroyAllWindows()
