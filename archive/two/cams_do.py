import subprocess
import time

def test_camera(cam_id):
    print(f"\n--- Testing Camera {cam_id} ---")
    try:
        # Capture a still image using libcamera-still
        output_file = f"camera_{cam_id}.jpg"
        result = subprocess.run([
            "libcamera-still",
            "-n",  # no preview
            "-t", "2000",  # 2 second delay
            "-o", output_file,
            "--camera", str(cam_id)
        ], check=True)
        print(f"✅ Camera {cam_id} captured image: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Camera {cam_id} failed to capture: {e}")

def main():
    print("📸 Starting dual-camera test on Raspberry Pi 5...")
    for cam in [0, 1]:
        test_camera(cam)
        time.sleep(1)
    print("\n🎉 Camera test complete! Check the output images.")

if __name__ == "__main__":
    main()