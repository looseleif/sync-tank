#!/usr/bin/env python3
"""Real Chromium portrait smoke and visual-regression screenshot."""

import argparse
import shutil
import struct
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
import sys

try:
    from playwright.sync_api import sync_playwright
except ImportError as exc:
    raise SystemExit("Playwright is required; install sync/requirements-browser.txt") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tank_manager import TankManagerApp, TankManagerHandler, TankManagerHTTPServer  # noqa: E402
from scripts.fake_tank_node import fixture_jpeg  # noqa: E402


def wait_for_server(url, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise SystemExit(f"Dashboard did not become ready at {url}")


def usable_screenshot(output):
    if not output.exists() or output.stat().st_size < 15_000:
        return False
    data = output.read_bytes()[:24]
    return len(data) == 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (1080, 1920)


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
        output.unlink(missing_ok=True)
        dashboard_url = f"http://127.0.0.1:{port}/?screenshot=1"
        try:
            wait_for_server(dashboard_url)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=chromium,
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-swiftshader"],
                )
                try:
                    page = browser.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=1)
                    page.goto(dashboard_url, wait_until="domcontentloaded", timeout=15_000)
                    page.wait_for_function(
                        "document.documentElement.dataset.screenshotReady === 'true'",
                        timeout=15_000,
                    )
                    page.screenshot(path=str(output), full_page=False)
                finally:
                    browser.close()
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
        if not usable_screenshot(output):
            raise SystemExit("Chromium did not produce a usable portrait screenshot")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="sync/tests/screenshots/see-sea-tv-1080x1920.png")
    args = parser.parse_args(); run(args.output)
