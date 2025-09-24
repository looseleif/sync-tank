import cv2
import os

def test_video_device(device_path):
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"[FAIL] {device_path} could not be opened.")
        return False

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print(f"[WARN] {device_path} opened but no valid frames.")
        cap.release()
        return False

    h, w, c = frame.shape
    print(f"[OK] {device_path} working! Resolution: {w}x{h}, Channels: {c}")
    cap.release()
    return True

def main():
    print("Scanning /dev/video* devices...\n")
    for i in range(50):  # check /dev/video0 to /dev/video9
        dev_path = f"/dev/video{i}"
        if os.path.exists(dev_path):
            test_video_device(dev_path)
        else:
            print(f"[SKIP] {dev_path} does not exist.")

if __name__ == "__main__":
    main()
