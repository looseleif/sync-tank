import cv2
import time
import threading
import requests
import os

# === CONFIG ===
INFERENCE_SERVER_URL = 'http://100.73.109.4:8000/infer'  # 🔧 Replace with your inference server IP
SEND_INTERVAL = 1  # seconds between sends

# Force-load Raspberry Pi CSI camera driver if present
os.system('sudo modprobe bcm2835-v4l2')

lock = threading.Lock()

def detect_cameras():
    working_cams = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"/dev/video{idx} opened and delivered frame.")
                working_cams.append((f"dev_video{idx}", cap))  # 🚀 SAFE camera name here!
            else:
                print(f"/dev/video{idx} opened but no valid frame. Skipping.")
                cap.release()
        else:
            print(f"Could not open /dev/video{idx}.")
    return working_cams

def send_frames(cam_name, cap):
    while True:
        with lock:
            ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from {cam_name}")
            time.sleep(SEND_INTERVAL)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print(f"Failed to encode frame from {cam_name}")
            time.sleep(SEND_INTERVAL)
            continue

        try:
            response = requests.post(
                INFERENCE_SERVER_URL,
                files={'image': jpeg.tobytes()},
                data={'camera': cam_name},  # ✅ Send safe name, e.g., 'dev_video0'
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

if __name__ == '__main__':
    cameras = detect_cameras()
    if not cameras:
        raise RuntimeError("❌ No working cameras found!")

    print("✅ Starting frame senders...")
    for cam_name, cap in cameras:
        threading.Thread(target=send_frames, args=(cam_name, cap), daemon=True).start()

    try:
        while True:
            time.sleep(10)
    finally:
        for _, cam in cameras:
            cam.release()
