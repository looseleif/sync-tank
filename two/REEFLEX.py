import cv2
import pygame
import numpy as np
import threading
import os
import smbus2
import time
import sys
from flask import Flask, Response, render_template_string

# Set correct HDMI display
os.environ["DISPLAY"] = ":0"

# Init I2C PCA9685
bus = smbus2.SMBus(1)
PCA9685_ADDR = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

bus.write_byte_data(PCA9685_ADDR, MODE1, 0x00)
time.sleep(0.005)
prescale_val = int(25000000.0 / (4096 * 50) - 1)
bus.write_byte_data(PCA9685_ADDR, MODE1, 0x10)
bus.write_byte_data(PCA9685_ADDR, PRESCALE, prescale_val)
bus.write_byte_data(PCA9685_ADDR, MODE1, 0x00)
time.sleep(0.005)
bus.write_byte_data(PCA9685_ADDR, MODE1, 0xA1)

def set_pwm(channel, on, off):
    reg = LED0_ON_L + 4 * channel
    bus.write_byte_data(PCA9685_ADDR, reg, on & 0xFF)
    bus.write_byte_data(PCA9685_ADDR, reg + 1, (on >> 8) & 0xFF)
    bus.write_byte_data(PCA9685_ADDR, reg + 2, off & 0xFF)
    bus.write_byte_data(PCA9685_ADDR, reg + 3, (off >> 8) & 0xFF)

def find_available_camera(max_index=10, timeout=2.0):
    """
    Scans camera indices up to max_index and returns the first available camera.
    Each camera has a timeout in seconds to prevent blocking.
    """
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        start_time = time.time()
        success, frame = False, None

        while time.time() - start_time < timeout:
            success, frame = cap.read()
            if success:
                break
            time.sleep(0.1)  # small wait before retry

        if success:
            cap.release()
            print(f"✅ Found working camera at index {idx}")
            return idx
        else:
            cap.release()
            print(f"⚠️ Camera index {idx} failed (timeout after {timeout} sec)")

    raise RuntimeError("❌ No working camera found within timeout limits!")

motors = [
    {"bank": 0, "midpoint": 400, "nudge": 30, "position": 400},
    {"bank": 1, "midpoint": 400, "nudge": 30, "position": 400},
    {"bank": 2, "midpoint": 400, "nudge": 30, "position": 400},
]

for motor in motors:
    set_pwm(motor["bank"], 0, motor["midpoint"])

step_size = 0.5
step_delay = 0.001
running = True

# Flask setup
app = Flask(__name__)
camera_index = find_available_camera()
cap = cv2.VideoCapture(camera_index)


HTML_PAGE = """
<html>
<head>
<title>ROV Control</title>
<style>
body { background-color: #222; color: #fff; text-align: center; font-family: sans-serif; }
button { margin: 5px; padding: 10px 20px; font-size: 18px; }
</style>
</head>
<body>
<h1>REEFLEX Controls</h1>
<img src="/video_feed" width="640" height="480">
<br>
<button onclick="sendCommand('w')">Forward (W)</button>
<button onclick="sendCommand('s')">Back (S)</button>
<button onclick="sendCommand('a')">Left (A)</button>
<button onclick="sendCommand('d')">Right (D)</button>
<button onclick="sendCommand('q')">Up (Q)</button>
<button onclick="sendCommand('e')">Down (E)</button>
<button onclick="sendCommand('stop')">Stop</button>
<script>
function sendCommand(cmd) {
  fetch('/control/' + cmd);
}
</script>
</body>
</html>
"""

def gen_frames():
    while running:
        success, frame = cap.read()
        if not success:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<cmd>')
def control(cmd):
    global motors
    idx_map = {'w': 0, 's': 0, 'a': 1, 'd': 1, 'q': 2, 'e': 2}
    idx = idx_map.get(cmd)
    if idx is None and cmd != 'stop':
        return 'Invalid command', 400

    if cmd == 'stop':
        for motor in motors:
            motor["position"] = motor["midpoint"]
            set_pwm(motor["bank"], 0, int(motor["midpoint"]))
    else:
        direction = 1 if cmd in ['w', 'a', 'q'] else -1
        motor = motors[idx]
        target = motor["position"] + motor["nudge"] * direction
        lower = motor["midpoint"] - motor["nudge"] * 5
        upper = motor["midpoint"] + motor["nudge"] * 5
        target = max(lower, min(upper, target))
        motor["position"] = target
        set_pwm(motor["bank"], 0, int(target))

    return 'OK'

def flask_thread():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

# Start Flask server in background
threading.Thread(target=flask_thread, daemon=True).start()
print("\n🌐 Web control ready at http://<pi-ip>:5000\n[Use W/A/S/D/Q/E/Stop buttons]\n")

# Initialize Pygame Fullscreen
pygame.init()
display_info = pygame.display.Info()
screen = pygame.display.set_mode((display_info.current_w, display_info.current_h), pygame.FULLSCREEN)
pygame.display.set_caption('ROV FPV')

# Frame Loop (local display)
try:
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (display_info.current_w, display_info.current_h))
        surf = pygame.surfarray.make_surface(np.rot90(frame_resized))

        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

except KeyboardInterrupt:
    running = False

cap.release()
pygame.quit()
print("Shutdown complete.")
