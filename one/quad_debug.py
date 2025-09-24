import cv2
import numpy as np
from picamera2 import Picamera2
import time

width, height = 640, 480

print("Initializing USB cameras...")
usb0 = cv2.VideoCapture('/dev/video1')
usb1 = cv2.VideoCapture('/dev/video11')
usb0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
usb0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
usb1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
usb1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
time.sleep(0.5)

print("Initializing PiCams...")
pi0 = Picamera2(0)
pi1 = Picamera2(1)
pi0.configure(pi0.create_preview_configuration(main={"size": (width, height)}))
pi1.configure(pi1.create_preview_configuration(main={"size": (width, height)}))
pi0.start()
pi1.start()
time.sleep(0.5)

print("Capturing frames...")
# Read from USB cams
ret0, frame_usb0 = usb0.read()
ret1, frame_usb1 = usb1.read()
print(f"USB0 status: {ret0}, USB1 status: {ret1}")

if not ret0:
    print("USB0 failed; black frame used")
    frame_usb0 = np.zeros((height, width, 3), dtype=np.uint8)
if not ret1:
    print("USB1 failed; black frame used")
    frame_usb1 = np.zeros((height, width, 3), dtype=np.uint8)

# Read from PiCams
frame_pi0 = pi0.capture_array()
frame_pi1 = pi1.capture_array()
print("Captured from PiCams")

frame_pi0 = cv2.cvtColor(cv2.resize(frame_pi0, (width, height)), cv2.COLOR_RGBA2BGR)
frame_pi1 = cv2.cvtColor(cv2.resize(frame_pi1, (width, height)), cv2.COLOR_RGBA2BGR)

# Combine frames
top = np.hstack((frame_usb0, frame_usb1))
bottom = np.hstack((frame_pi0, frame_pi1))
grid = np.vstack((top, bottom))

print("Saving preview image as output.jpg...")
cv2.imwrite("output.jpg", grid)

# Clean up
usb0.release()
usb1.release()
pi0.stop()
pi1.stop()
print("Done.")