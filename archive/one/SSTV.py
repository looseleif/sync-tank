import os
import cv2
import time
import threading
import requests
from flask import Flask, Response, render_template_string, redirect, url_for
from picamera2 import Picamera2
import pyautogui

# === CONFIG ===
INFERENCE_SERVER_URL = 'http://100.73.109.4:8000/infer'
SEND_INTERVAL = 1  # seconds between sends
SHOW_ON_MONITOR = True  # Set True to show fullscreen local monitor feed

# Load Raspberry Pi CSI camera driver
os.system('sudo modprobe bcm2835-v4l2')

app = Flask(__name__)
lock = threading.Lock()
current_camera_idx = 0

# Global detection counts per camera
detection_counts = []

def move_mouse_offscreen():
    try:
        pyautogui.FAILSAFE = False
        width, height = pyautogui.size()
        pyautogui.moveTo(width - 1, height - 1)
        print(f"🖱️ Mouse cursor moved to bottom-right ({width -1}, {height -1}).")
    except Exception as e:
        print(f"⚠️ Could not move mouse: {e}")
        
# === Detect USB cameras ===
def detect_usb_cameras():
    usb_cams = []
    max_index = 10  # test first 10 /dev/video*
    
    def try_open(idx):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ USB Camera /dev/video{idx} opened.")
                usb_cams.append((f"dev_video{idx}", cap, 'usb'))
            else:
                cap.release()

    threads = []
    for idx in range(max_index):
        t = threading.Thread(target=try_open, args=(idx,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=1)

    return usb_cams

# === Initialize PiCameras ===
def init_picameras():
    picams = []
    for idx in range(2):
        try:
            cam = Picamera2(idx)
            config = cam.create_video_configuration(main={"size": (640, 480)})
            cam.configure(config)
            cam.start()
            print(f"✅ PiCamera {idx} started.")
            picams.append((f"picam{idx}", cam, 'picam'))
        except Exception as e:
            print(f"❌ Failed to start PiCamera {idx}: {e}")
    return picams

# === Unified frame read ===
def read_frame(cam_obj, cam_type):
    if cam_type == 'usb':
        ret, frame = cam_obj.read()
        if not ret:
            return None
        return frame
    elif cam_type == 'picam':
        return cam_obj.capture_array()

# === Inference sending ===
def send_frames(cam_name, cam_obj, cam_type, cam_idx):
    global detection_counts
    while True:
        with lock:
            frame = read_frame(cam_obj, cam_type)
        if frame is None:
            print(f"⚠️ Failed to grab frame from {cam_name}")
            time.sleep(SEND_INTERVAL)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print(f"⚠️ Failed to encode frame from {cam_name}")
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
                detections = result.get('detections', [])
                detection_count = len(detections) if detections else 0
                detection_counts[cam_idx] = detection_count
                print(f"{cam_name} detections: {detections}")
            else:
                print(f"⚠️ Inference server error {cam_name}: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error sending {cam_name} frame: {e}")

        time.sleep(SEND_INTERVAL)

# === Find most active camera index ===
def get_most_active_camera_idx():
    if len(detection_counts) != len(cameras):
        return 0
    max_count = max(detection_counts)
    if max_count == 0:
        return 0
    return detection_counts.index(max_count)

# === Flask frame generator ===
def gen_frames(idx):
    cam_name, cam_obj, cam_type = cameras[idx]
    while True:
        with lock:
            frame = read_frame(cam_obj, cam_type)
        if frame is None:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    camera_list = [(i, cam[0]) for i, cam in enumerate(cameras)]
    return render_template_string(PAGE_TEMPLATE,
                                  camera_idx=current_camera_idx,
                                  camera_name=cameras[current_camera_idx][0],
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
        print(f"🔀 Switched to {cameras[cam_idx][0]} via web")
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
    usb_cameras = detect_usb_cameras()
    picam_cameras = init_picameras()
    cameras = usb_cameras + picam_cameras

    if not cameras:
        raise RuntimeError("❌ No working cameras found!")

    detection_counts = [0] * len(cameras)

    # Start inference sender threads
    for idx, (cam_name, cam_obj, cam_type) in enumerate(cameras):
        threading.Thread(target=send_frames, args=(cam_name, cam_obj, cam_type, idx), daemon=True).start()

    # Start Flask web thread
    threading.Thread(target=flask_thread, daemon=True).start()

    print("\n✅ Multi-Cam streamer + inference sender running!")
    print("🔗 Web UI at: http://<raspi-ip>:5000/")

    try:
        if SHOW_ON_MONITOR:
            move_mouse_offscreen()  # hide/move mouse

            window_name = 'Local Camera Feed'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            while True:
                with lock:
                    current_camera_idx = get_most_active_camera_idx()
                    cam_name, cam_obj, cam_type = cameras[current_camera_idx]
                    frame = read_frame(cam_obj, cam_type)
                if frame is not None:
                    cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.01)
        else:
            while True:
                time.sleep(1)
    finally:
        for cam_name, cam_obj, cam_type in usb_cameras:
            cam_obj.release()
        if SHOW_ON_MONITOR:
            cv2.destroyAllWindows()
