import os
import cv2
import time
import base64
import json
import subprocess
import requests
from ultralytics import YOLO

# === CONFIG ===
IMAGE_DIR = '/home/chase/Projects/sync-tank/deep-link/dataset/train/images'
OLLAMA_MODEL = "llava:7b"
OLLAMA_PORT = 11500  # update to match Ollama server on Windows
OLLAMA_TIMEOUT = 120
PROMPT_TEMPLATE = "You are an expert aquarium guide. Describe the fish and make it short in response, KEEP REPLY 10 WORDS OR LESS"
CONFIDENCE_THRESHOLD = 0.8

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

# === MAIN PROCESS ===
def process_images():
    model = YOLO('weights.pt')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🔍 Found {len(image_files)} images in {IMAGE_DIR}")

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ Could not read {img_name}")
            continue

        results = model(frame)
        detections = []
        for box in results[0].boxes:
            conf = float(box.conf.cpu().numpy())
            label = results[0].names[int(box.cls.cpu().numpy())]
            if conf >= CONFIDENCE_THRESHOLD:
                detections.append({'label': label, 'confidence': round(conf, 3)})

        print(f"📸 {img_name} detections: {detections}")

        if any(d['label'] == 'fish' for d in detections):
            summary = send_to_ollama(img_path, PROMPT_TEMPLATE + ", ".join([d['label'] for d in detections]))
            if summary:
                print(f"💬 Guide summary: {summary}")
        time.sleep(2)  # optional: avoid spamming too fast

if __name__ == '__main__':
    print("✅ Starting local image directory test...")
    process_images()
