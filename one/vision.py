import cv2

cap = cv2.VideoCapture(0)  # Using /dev/video0 for your endoscopic cam

if not cap.isOpened():
    print("[❌] Cannot open camera")
    exit()

# Create fullscreen window
cv2.namedWindow("Endoscopic View", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Endoscopic View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[⚠️] Failed to grab frame")
        break

    cv2.imshow("Endoscopic View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()