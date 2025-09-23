import cv2
import time
import threading
import requests
from flask import Flask, Response, render_template_string, redirect, url_for
import os

# === CONFIG ===
INFERENCE_SERVER_URL = 'http://100.73.109.4:8000/infer'  # Replace with your inference server IP
SEND_INTERVAL = 1  # seconds between sends
SHOW_ON_MONITOR = False  # Set True to show fullscreen local monitor feed

# Force-load Raspberry Pi CSI camera driver if present
os.system('sudo modprobe bcm2835-v4l2')

app = Flask(__name__)
lock = threading.Lock()
current_camera_idx = 0

def detect_cameras():
    working_cams = []
    for idx in range(10):
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

def send_frames(cam_name, cap):
    while True:
        with lock:
            ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to grab frame from {cam_name}")
            time.sleep(SEND_INTERVAL)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print(f"❌ Failed to encode frame from {cam_name}")
            time.sleep(SEND_INTERVAL)
            continue

        try:
            response = requests.post(
                INFERENCE_SERVER_URL,
                files={'image': jpeg.tobytes()},
                data={'camera': cam_name},
                timeout=5
            )
            if response.ok:
                result = response.json()
                print(f"{cam_name} detections: {result.get('detections')}")
            else:
                print(f"❌ Inference server error {cam_name}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending {cam_name} frame: {e}")

        time.sleep(SEND_INTERVAL)

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

if __name__ == '__main__':
    cameras = detect_cameras()
    if not cameras:
        raise RuntimeError("❌ No working cameras found!")

    # Start inference sender threads
    for idx, cap in cameras:
        cam_name = f"dev_video{idx}"
        threading.Thread(target=send_frames, args=(cam_name, cap), daemon=True).start()

    # Start Flask web thread
    threading.Thread(target=flask_thread, daemon=True).start()

    print("\n✅ Multi-Cam streamer + inference sender running!")
    print("🔗 Web UI at: http://<raspi-ip>:5000/")
    if SHOW_ON_MONITOR:
        print("🖥️ Local monitor showing fullscreen feed. Press 'q' to quit.")
    else:
        print("🖥️ Local monitor display is DISABLED. Press CTRL+C to quit.")

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
