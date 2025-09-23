import cv2
import pygame
import numpy as np
import threading
import os
import smbus2
import time
import sys
import termios
import tty
import select
from flask import Flask, Response

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
cap = cv2.VideoCapture(0)

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
    return '<h1>Pi Camera Stream</h1><p>Go to <a href="/video_feed">/video_feed</a></p>'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def flask_thread():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

# Start Flask server in background
threading.Thread(target=flask_thread, daemon=True).start()

# Keyboard Listener
def keyboard_control():
    global running
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while running:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == '\x1b':
                    running = False
                    break
                idx = {'w': 0, 's': 0, 'a': 1, 'd': 1, 'q': 2, 'e': 2}.get(key)
                if idx is None:
                    continue
                direction = 1 if key in ['w', 'a', 'q'] else -1
                motor = motors[idx]
                target = motor["position"] + motor["nudge"] * direction
                lower = motor["midpoint"] - motor["nudge"] * 5
                upper = motor["midpoint"] + motor["nudge"] * 5
                target = max(lower, min(upper, target))

                while abs(motor["position"] - target) >= step_size:
                    motor["position"] += step_size if motor["position"] < target else -step_size
                    set_pwm(motor["bank"], 0, int(motor["position"]))
                    time.sleep(step_delay)

                motor["position"] = target
                set_pwm(motor["bank"], 0, int(target))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# Initialize Pygame Fullscreen
pygame.init()
display_info = pygame.display.Info()
screen = pygame.display.set_mode((display_info.current_w, display_info.current_h), pygame.FULLSCREEN)
pygame.display.set_caption('ROV FPV')

# Start keyboard control thread
threading.Thread(target=keyboard_control, daemon=True).start()
print("\n[WASD] Direction | [Q/E] Up/Down | [ESC] Quit\n")

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