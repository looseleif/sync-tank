from flask import Flask, Response
import cv2

app = Flask(__name__)

# Update device mapping if needed
cams = {
    'arducam': '/dev/video0',
    'endoscope': '/dev/video2'
}

# Open video streams
cap_arducam = cv2.VideoCapture(cams['arducam'])
cap_endo = cv2.VideoCapture(cams['endoscope'])

def generate_frames(cap):
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <h1>Synk Tank Feeds</h1>
    <h2>Arducam Feed:</h2>
    <img src="/video_arducam">
    <h2>Endoscope Feed:</h2>
    <img src="/video_endo">
    """

@app.route('/video_arducam')
def video_arducam():
    return Response(generate_frames(cap_arducam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_endo')
def video_endo():
    return Response(generate_frames(cap_endo),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
