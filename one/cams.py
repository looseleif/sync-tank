import cv2
import numpy as np
from picamera2 import Picamera2
import time

# USB camera setup
usb0 = cv2.VideoCapture(0)
usb1 = cv2.VideoCapture(1)

# Pi camera setup
picam0 = Picamera2(0)
picam1 = Picamera2(1)

picam0.configure(picam0.create_preview_configuration())
picam1.configure(picam1.create_preview_configuration())

picam0.start()
picam1.start()
time.sleep(1)  # Let them warm up

while True:
    _, frame_usb0 = usb0.read()
    _, frame_usb1 = usb1.read()
    
    frame_pi0 = picam0.capture_array()
    frame_pi1 = picam1.capture_array()

    # Resize all to same dimensions
    size = (320, 240)
    frame_usb0 = cv2.resize(frame_usb0, size)
    frame_usb1 = cv2.resize(frame_usb1, size)
    frame_pi0 = cv2.resize(frame_pi0, size)
    frame_pi1 = cv2.resize(frame_pi1, size)

    # Stack frames
    top_row = np.hstack((frame_usb0, frame_usb1))
    bottom_row = np.hstack((frame_pi0, frame_pi1))
    grid = np.vstack((top_row, bottom_row))

    # Show to display
    cv2.imshow("2x2 Camera Grid", grid)
    if cv2.waitKey(1) == 27:
        break

usb0.release()
usb1.release()
cv2.destroyAllWindows()