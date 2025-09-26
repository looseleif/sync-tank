import os
import cv2
import time
import threading
import base64
import subprocess
import requests
import numpy as np
import logging
from flask import Flask, request, Response, jsonify
from ultralytics import YOLO

# === CONFIG ===
OLLAMA_MODEL = "llava:7b"
OLLAMA_PORT = 11500
OLLAMA_TIMEOUT = 120  # seconds
PROMPT_TEMPLATE = "You are an expert aquarium guide. Describe the scene in front of you and make it funny but keep it to 10 words or less: "
CONFIDENCE_THRESHOLD = 0.8
SEND_INTERVAL = 10  # seconds between LLM sends
SUPPRESS_FLASK_LOGS = True

# === OPTIONAL: suppress Flask request logs ===
if SUPPRESS_FLASK_LOGS:
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

# === AUTO-DETECT WINDOWS HOST IP ===
def get_windows_host_ip():
    try:
        result = subprocess.run(["ip", "route"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "default via" in line:
                return line.split()[2]
    except Exception as e:
        print(f"[ERROR] Failed to detect Windows host IP: {e}")
    return None

# === SETUP ===
app = Flask(__name__)
model = YOLO('weights.pt')
# model = YOLO('weights.pt')
latest_frames = {}
lock = threading.Lock()
last_sent_time = 0

def send_to_ollama(image_path, prompt):
    win_ip = get_windows_host_ip()
    if not win_ip:
        print("[ERROR] No Windows host IP detected.")
        return None
    url = f"http://{win_ip}:{OLLAMA_PORT}/api/generate"
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"num_predict": 50}
    }
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        text = result.get('response', '').strip()
        print(f"[INFO] Ollama summary: {text}")
        return text
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return None

@app.route('/infer', methods=['POST'])
def infer():
    global last_sent_time
    if 'image' not in request.files or 'camera' not in request.form:
        return jsonify({'error': 'Missing image or camera field'}), 400

    cam_name = request.form['camera']
    npimg = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    detection_summary = {}
    for box in results[0].boxes:
        conf = float(box.conf.cpu().numpy().item())
        label_idx = int(box.cls.cpu().numpy().item())
        label = results[0].names[label_idx]
        if conf >= CONFIDENCE_THRESHOLD:
            detection_summary[label] = detection_summary.get(label, 0) + 1

    with lock:
        latest_frames[cam_name] = {'frame': annotated_frame, 'detections': detection_summary}

    if detection_summary:
        det_str = ', '.join(f"{label} ({count})" for label, count in detection_summary.items())
        print(f"[INFO] Camera: {cam_name} | Detected: {det_str}")
    else:
        print(f"[INFO] Camera: {cam_name} | No detections")

    now = time.time()
    summary = None

    if 'fish' in detection_summary and (now - last_sent_time) > SEND_INTERVAL:
        tmp_path = f"/tmp/{cam_name}_snapshot.jpg"
        cv2.imwrite(tmp_path, frame)
        summary = send_to_ollama(tmp_path, PROMPT_TEMPLATE + ', '.join(detection_summary.keys()))
        last_sent_time = now

    # Always include caption in dashboard
    with lock:
        latest_frames[cam_name]['caption'] = summary or ''

    return jsonify({'detections': detection_summary, 'caption': summary or ''})

# === STREAM ROUTE ===
def gen_stream(cam_name):
    while True:
        with lock:
            frame_data = latest_frames.get(cam_name)
        if frame_data:
            ret, buffer = cv2.imencode('.jpg', frame_data['frame'])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route('/stream/<cam_name>')
def stream(cam_name):
    return Response(gen_stream(cam_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    with lock:
        cam_names = list(latest_frames.keys())
    html = """
    <html><head><title>SyncTank Deeplink</title></head>
    <body style='background:#111;color:#fff; font-family:sans-serif;'>
    <h1>SyncTank Deeplink</h1>
    <div style='display:flex; flex-wrap:wrap; justify-content:center;'>
    """
    for cam in cam_names:
        caption = latest_frames[cam].get('caption', '')
        det_summary = latest_frames[cam].get('detections', {})
        det_str = ', '.join(f"{label} ({count})" for label, count in det_summary.items()) or "No detections"
        html += f"""
        <div style='margin:10px; text-align:center;'>
            <h3>{cam}</h3>
            <img src='/stream/{cam}' width='400' style='border:2px solid #555;'><br>
            <div style='margin-top:5px; font-size:14px; color:#0f0;'>
                Detections: {det_str}<br>
            </div>
        </div>
        """
    html += "</div></body></html>"
    return html

if __name__ == '__main__':
    print("✅ Inference + Ollama server running at http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)
