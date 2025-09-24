import os
import cv2
import threading
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

app = Flask(__name__)

# Initialize cameras
picam0 = Picamera2(0)
picam1 = Picamera2(1)

config0 = picam0.create_video_configuration(main={"size": (640, 480)})
config1 = picam1.create_video_configuration(main={"size": (640, 480)})

picam0.configure(config0)
picam1.configure(config1)

picam0.start()
picam1.start()

lock = threading.Lock()

# HTML page template
PAGE_TEMPLATE = """
<html>
<head>
    <title>Pi Camera Streams</title>
    <style>
        body { background-color: #111; color: #fff; text-align: center; font-family: sans-serif; }
        img { border: 2px solid #555; margin: 10px; }
    </style>
</head>
<body>
    <h1>Pi Camera Streams</h1>
    <div>
        <h2>Camera 0</h2>
        <img src="{{ url_for('video_feed', cam_id=0) }}" width="640" height="480">
    </div>
    <div>
        <h2>Camera 1</h2>
        <img src="{{ url_for('video_feed', cam_id=1) }}" width="640" height="480">
    </div>
</body>
</html>
"""

def gen_frames(camera):
    while True:
        with lock:
            frame = camera.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(PAGE_TEMPLATE)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    if cam_id == 0:
        return Response(gen_frames(picam0), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif cam_id == 1:
        return Response(gen_frames(picam1), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid camera ID", 404

if __name__ == '__main__':
    print("✅ Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
