import os
import cv2
import time
import threading
import requests
import numpy as np
from flask import Flask, Response, render_template_string, redirect, url_for
from picamera2 import Picamera2

# CONFIG
INFERENCE_SERVER_URL = 'http://100.73.109.4:8000/infer'
SEND_INTERVAL = 1  # seconds
SHOW_ON_MONITOR = True
MIN_DISPLAY_TIME = 10  # seconds per camera before switching
CAPTION_DISPLAY_DURATION = 10  # seconds to keep showing old caption

app = Flask(__name__)
lock = threading.Lock()
current_camera_idx = 0
last_switch_time = 0

USB_CAM_INDEXES = [0, 2, 4]  # Only known working USB cams

class CameraHandler:
    def __init__(self, name, cam_obj, cam_type):
        self.name = name
        self.cam_obj = cam_obj
        self.cam_type = cam_type
        self.latest_frame = None
        self.latest_caption = ""
        self.latest_caption_time = 0  # timestamp when caption was last updated
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def capture_loop(self):
        while self.running:
            frame = self.read_frame()
            if frame is not None and frame.size > 0:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.01)

    def read_frame(self):
        if self.cam_type == 'usb':
            ret, frame = self.cam_obj.read()
            return frame if ret else None
        elif self.cam_type == 'picam':
            return self.cam_obj.capture_array()

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def release(self):
        self.running = False
        if self.cam_type == 'usb':
            self.cam_obj.release()
        elif self.cam_type == 'picam':
            self.cam_obj.close()

def detect_usb_cameras():
    usb_cams = []
    for idx in USB_CAM_INDEXES:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[INFO] USB camera /dev/video{idx} confirmed working")
                usb_cams.append(CameraHandler(f"dev_video{idx}", cap, 'usb'))
            else:
                print(f"[WARN] /dev/video{idx} opened but no frames")
                cap.release()
        else:
            print(f"[WARN] /dev/video{idx} could not open")
    return usb_cams

def init_picameras():
    picams = []
    for idx in range(2):
        try:
            cam = Picamera2(idx)
            config = cam.create_video_configuration(main={"size": (640, 480)})
            cam.configure(config)
            cam.start()
            print(f"[INFO] CSI camera picam{idx} started")
            picams.append(CameraHandler(f"picam{idx}", cam, 'picam'))
        except Exception as e:
            print(f"[WARN] Failed to init picam{idx}: {e}")
    return picams

def send_frames(cam_handler, cam_idx):
    while True:
        frame = cam_handler.get_frame()
        if frame is None:
            time.sleep(SEND_INTERVAL)
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(SEND_INTERVAL)
            continue
        try:
            response = requests.post(
                INFERENCE_SERVER_URL,
                files={'image': jpeg.tobytes()},
                data={'camera': cam_handler.name},
                timeout=5
            )
            if response.ok:
                result = response.json()
                detections = result.get('detections', [])
                caption = result.get('caption', '')
                if caption:  # only update if non-empty
                    cam_handler.latest_caption = caption
                    cam_handler.latest_caption_time = time.time()
                print(f"[INFO] {cam_handler.name}: {len(detections)} detections, caption: {caption}")
            else:
                print(f"[WARN] {cam_handler.name}: inference server error {response.status_code}")
        except Exception as e:
            print(f"[ERROR] {cam_handler.name}: sending frame failed: {e}")
        time.sleep(SEND_INTERVAL)

def overlay_caption(frame, caption, cam_handler):
    now = time.time()
    if now - cam_handler.latest_caption_time <= CAPTION_DISPLAY_DURATION:
        height, width, _ = frame.shape
        font_scale = 0.5  # smaller text
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size = cv2.getTextSize(caption, font, font_scale, thickness)[0]
        x = 10  # left margin
        y = height - 10  # bottom margin

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - 30), (width, height), (0, 0, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw text
        cv2.putText(frame, caption, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return frame

def gen_frames(cam_handler):
    while True:
        frame = cam_handler.get_frame()
        if frame is None:
            continue
        caption = cam_handler.latest_caption
        frame = overlay_caption(frame, caption, cam_handler)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    camera_list = [(i, cam.name) for i, cam in enumerate(cameras)]
    return render_template_string(PAGE_TEMPLATE,
                                  camera_idx=current_camera_idx,
                                  camera_name=cameras[current_camera_idx].name,
                                  camera_list=camera_list)

@app.route('/video_feed/<int:cam_idx>')
def video_feed(cam_idx):
    if 0 <= cam_idx < len(cameras):
        return Response(gen_frames(cameras[cam_idx]),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid camera index", 404

@app.route('/switch/<int:cam_idx>')
def switch_camera(cam_idx):
    global current_camera_idx, last_switch_time
    if 0 <= cam_idx < len(cameras):
        current_camera_idx = cam_idx
        last_switch_time = time.time()
        print(f"[INFO] Switched to camera {cameras[cam_idx].name}")
    return redirect(url_for('index'))

def flask_thread():
    app.run(host='0.0.0.0', port=5000, threaded=True)

PAGE_TEMPLATE = """
<html>
<head>
    <title>SyncTank Streamer</title>
    <style>
        body { background:#111; color:#fff; text-align:center; font-family:sans-serif; }
        button { margin:5px; padding:10px 20px; font-size:16px; }
        img { margin-top:10px; border:2px solid #555; max-width:90vw; height:auto; }
    </style>
</head>
<body>
    <h1>See Sea TV</h1>
    <img src="{{ url_for('video_feed', cam_idx=camera_idx) }}">
    <p>Currently showing: <b>{{ camera_name }}</b></p>
    {% for idx, name in camera_list %}
        <a href="{{ url_for('switch_camera', cam_idx=idx) }}">
            <button>Switch to {{ name }}</button>
        </a>
    {% endfor %}
</body>
</html>
"""

if __name__ == '__main__':
    usb_cameras = detect_usb_cameras()
    picam_cameras = init_picameras()
    cameras = usb_cameras + picam_cameras

    if not cameras:
        raise RuntimeError("[FATAL] No working cameras found!")

    for idx, cam_handler in enumerate(cameras):
        threading.Thread(target=send_frames, args=(cam_handler, idx), daemon=True).start()

    threading.Thread(target=flask_thread, daemon=True).start()

    last_switch_time = time.time()
    print("[INFO] SyncTank streamer running at http://<raspi-ip>:5000/")

    try:
        if SHOW_ON_MONITOR:
            cv2.namedWindow('SyncTank', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('SyncTank', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            while True:
                now = time.time()
                if now - last_switch_time >= MIN_DISPLAY_TIME:
                    current_camera_idx = (current_camera_idx + 1) % len(cameras)
                    last_switch_time = now
                frame = cameras[current_camera_idx].get_frame()
                if frame is not None:
                    caption = cameras[current_camera_idx].latest_caption
                    frame = overlay_caption(frame, caption, cameras[current_camera_idx])
                    cv2.imshow('SyncTank', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.05)
        else:
            while True:
                time.sleep(1)
    finally:
        for cam_handler in cameras:
            cam_handler.release()
        cv2.destroyAllWindows()
