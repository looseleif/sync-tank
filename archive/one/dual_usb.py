import cv2
import numpy as np

# Adjust these if your devices differ
cam0 = cv2.VideoCapture('/dev/video1')
cam1 = cv2.VideoCapture('/dev/video11')

# Set resolution
width, height = 640, 480
cam0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Press 'q' to quit.")

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if not ret0:
        frame0 = np.zeros((height, width, 3), dtype=np.uint8)
    if not ret1:
        frame1 = np.zeros((height, width, 3), dtype=np.uint8)

    # Resize if needed (safety)
    frame0 = cv2.resize(frame0, (width, height))
    frame1 = cv2.resize(frame1, (width, height))

    stacked = np.hstack((frame0, frame1))
    cv2.imshow("USB Cameras", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam0.release()
cam1.release()
cv2.destroyAllWindows()