import cv2
import numpy as np
from picamera2 import Picamera2
import time

# === Setup USB cameras ===
usb0 = cv2.VideoCapture('/dev/video0')
usb1 = cv2.VideoCapture('/dev/video18')
width, height = 640, 480
usb0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
usb0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
usb1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
usb1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# === Setup PiCamera2 cameras ===
pi0 = Picamera2(0)
pi1 = Picamera2(1)

print("Configuring PiCams")
pi0.configure(pi0.create_preview_configuration(main={"size": (width, height)}))
pi1.configure(pi1.create_preview_configuration(main={"size": (width, height)}))

print("Starting PiCams")
pi0.start()
print("Started pi0")
pi1.start()
print("Started pi1")

print("Setting up USB cams")

print("Press 'q' to quit.")

try:
    while True:
        # --- Read from USB cameras ---
        ret0, frame_usb0 = usb0.read()
        ret1, frame_usb1 = usb1.read()

        if not ret0:
            frame_usb0 = np.zeros((height, width, 3), dtype=np.uint8)
        if not ret1:
            frame_usb1 = np.zeros((height, width, 3), dtype=np.uint8)

        # --- Read from PiCamera2 ---
        frame_pi0 = pi0.capture_array()
        frame_pi1 = pi1.capture_array()

        frame_pi0 = cv2.cvtColor(cv2.resize(frame_pi0, (width, height)), cv2.COLOR_RGBA2BGR)
        frame_pi1 = cv2.cvtColor(cv2.resize(frame_pi1, (width, height)), cv2.COLOR_RGBA2BGR)

        # --- Stack the 4 frames into 2x2 grid ---
        top_row = np.hstack((frame_usb0, frame_usb1))
        bottom_row = np.hstack((frame_pi0, frame_pi1))
        full_grid = np.vstack((top_row, bottom_row))

        # Show combined view
        cv2.imshow("2x2 Camera Feed", full_grid)

        print("USB0:", frame_usb0.shape, "USB1:", frame_usb1.shape)
        print("Pi0:", frame_pi0.shape, "Pi1:", frame_pi1.shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# === Cleanup ===
usb0.release()
usb1.release()
pi0.stop()
pi1.stop()
cv2.destroyAllWindows()