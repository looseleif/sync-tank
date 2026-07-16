#!/usr/bin/env python3
import sys, time, cv2
from picamera2 import Picamera2

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
W, H, Q = 1280, 720, 90

p2 = Picamera2(camera_num=cam)
cfg = p2.create_still_configuration({"size": (W, H)})
p2.configure(cfg)
p2.start()
time.sleep(0.15)  # small settle
rgb = p2.capture_array()
p2.stop(); p2.close()

bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), Q])
open(f"/tmp/picam_cam{cam}.jpg", "wb").write(jpg)
print(f"Wrote /tmp/picam_cam{cam}.jpg")
