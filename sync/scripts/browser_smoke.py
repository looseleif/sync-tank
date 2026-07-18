#!/usr/bin/env python3
"""Real Chromium portrait smoke and visual-regression screenshot."""

import argparse
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tank_manager import TankManagerApp, TankManagerHandler, TankManagerHTTPServer  # noqa: E402
from scripts.fake_tank_node import fixture_jpeg  # noqa: E402


def run(output):
    chromium = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if not chromium:
        raise SystemExit("Chromium is required for the browser smoke")
    with tempfile.TemporaryDirectory(prefix="sync-browser-") as storage:
        app = TankManagerApp(storage)
        app.register_node({"node_id": "fake-1", "tank_ids": ["tank-1"], "status": "online"})
        app.register_node({"node_id": "fake-2", "tank_ids": ["tank-2"], "status": "online"})
        server = TankManagerHTTPServer(("127.0.0.1", 0), TankManagerHandler, app)
        port = server.server_address[1]
        cameras = []
        for index in range(8):
            tank_number = 1 + index % 2
            camera_id = f"browser-cam-{index}"
            cameras.append({"camera_id": camera_id, "node_id": f"fake-{tank_number}", "tank_id": f"tank-{tank_number}",
                            "status": "online", "source_type": "usb_camera", "camera_type": "lighthouse_cam" if index == 0 else "endoscope_cam",
                            "role_locked": True, "label": "Raydar" if index == 0 else f"Reel {index}",
                            "stream_url": f"http://127.0.0.1:{port}/uploads/{camera_id}/latest.jpg"})
        app.register_cameras(cameras)
        for index, camera in enumerate(cameras):
            app.handle_frame_upload({"camera_id": camera["camera_id"], "image_bytes": fixture_jpeg("fish" if index == 0 else "decorations", index)})
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        profile = Path(storage) / "chromium-profile"
        launcher_flags = ["--no-memcheck"] if Path(chromium).name == "chromium" else []
        command = [chromium, *launcher_flags, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                   f"--user-data-dir={profile}", "--window-size=1080,1920", "--virtual-time-budget=7000",
                   f"--screenshot={output}", f"http://127.0.0.1:{port}"]
        try:
            subprocess.run(command, check=True, timeout=45)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
        if not output.exists() or output.stat().st_size < 10_000:
            raise SystemExit("Chromium did not produce a usable portrait screenshot")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="sync/tests/screenshots/see-sea-tv-1080x1920.png")
    args = parser.parse_args(); run(args.output)
