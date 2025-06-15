import os
import cv2
import time
import threading
import base64
import json
import subprocess
import requests
from flask import Flask, request, Response, jsonify
from ultralytics import YOLO

# === CONFIG ===
OLLAMA_MODEL = "llava:7b"
OLLAMA_PORT = 11500
OLLAMA_TIMEOUT = 120  # seconds
PROMPT_TEMPLATE = "You are an expert aquarium guide. Describe the fish seen: "
CONFIDENCE_THRESHOLD = 0.8
SEND_INTERVAL = 10  # seconds between LLM sends

# === AUTO-DETECT WINDOWS HOST IP ===
def get_windows_host_ip():
    try:
        result = subprocess.run(["ip", "route"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "default via" in line:
                return line.split()[2]
    except Exception as e:
        print(f"❌ Failed to detect Windows host IP: {e}")
    return None

# === SETUP ===
app = Flask(__name__)
model = YOLO('weights.pt')
latest_frames = {}
lock = threading.Lock()
last_sent_time = 0

# === SEND IMAGE TO OLLAMA ===
def send_to_ollama(image_path, prompt):
    win_ip = get_windows_host_ip()
    if not win_ip:
        print("❌ No Windows host IP detected.")
        return None
    url = f"http://{win_ip}:{OLLAMA_PORT}/api/generate"
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        text = result.get('response', '').strip()
        print(f"🧠 Ollama says: {text}")
        return text
    except Exception as e:
        print(f"❌ Ollama request failed: {e}")
        return None

# === INFERENCE ROUTE ===
@app.route('/infer', methods=['POST'])
def infer():
    global last_sent_time
    if 'image' not in request.files or 'camera' not in request.form:
        return jsonify({'error': 'Missing image or camera field'}), 400
    cam_name = request.form['camera']
    npimg = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    results = model(frame)
    annotated_frame = results[0].plot()
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf.cpu().numpy())
        label = results[0].names[int(box.cls.cpu().numpy())]
        if conf >= CONFIDENCE_THRESHOLD:
            detections.append({'label': label, 'confidence': round(conf, 3)})
            
    with lock:
        latest_frames[cam_name] = {'frame': annotated_frame, 'detections': detections}

    # Save temp image and send to Ollama every SEND_INTERVAL if fish detected
    now = time.time()
    if any(d['label'] == 'fish' for d in detections) and (now - last_sent_time) > SEND_INTERVAL:
        tmp_path = f"/tmp/{cam_name}_snapshot.jpg"
        cv2.imwrite(tmp_path, frame)
        summary = send_to_ollama(tmp_path, PROMPT_TEMPLATE + ", ".join([d['label'] for d in detections]))
        if summary:
            print(f"💬 Guide summary: {summary}")
        last_sent_time = now

    return jsonify({'detections': detections})

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
    <html><head><title>SyncTank Dashboard</title></head><body style='background:#111;color:#fff;'>
    <h1>🐟 SyncTank Dashboard</h1><div style='display:flex;flex-wrap:wrap;'>
    """
    for cam in cam_names:
        html += f"<div style='margin:10px'><h3>{cam}</h3><img src='/stream/{cam}' width='400'></div>"
    html += "</div></body></html>"
    return html

if __name__ == '__main__':
    print("✅ Inference + Ollama server running at http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, threaded=True)

