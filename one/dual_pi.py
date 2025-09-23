from picamera2 import Picamera2, Preview
import time

# Camera 0
cam0 = Picamera2(0)
cam0.configure(cam0.create_preview_configuration())
cam0.start_preview(Preview.QT)
cam0.start()

# Camera 1
cam1 = Picamera2(1)
cam1.configure(cam1.create_preview_configuration())
cam1.start_preview(Preview.QT)
cam1.start()

print("Both PiCams running — press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    cam0.stop()
    cam1.stop()