#!/usr/bin/env python3
"""Accelerated offline soak for feed rotation, persistence, and captures."""

import argparse
import tempfile
import time
import tracemalloc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tank_manager import TankManagerApp  # noqa: E402
from scripts.fake_tank_node import fixture_jpeg  # noqa: E402


def run(duration, cycles_per_second=20, accelerated=True):
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="sync-soak-") as storage:
        app = TankManagerApp(storage)
        app.tanks = {f"tank-{number}": app._default_tank(f"tank-{number}") for number in (1, 2)}
        cameras = [{"camera_id": f"cam-{index}", "tank_id": f"tank-{1 + index % 2}", "status": "online", "source_type": "usb_camera"} for index in range(8)]
        app.register_cameras(cameras)
        start = time.monotonic(); cycles = 0
        target_cycles = max(1, int(duration)) if accelerated else None
        while (cycles < target_cycles) if accelerated else (time.monotonic() - start < duration):
            camera = cameras[cycles % len(cameras)]
            app.handle_frame_upload({"camera_id": camera["camera_id"], "image_bytes": fixture_jpeg("fish" if cycles % 7 == 0 else "empty", cycles)})
            app.get_layout()
            if cycles % 50 == 0:
                app.capture_sighting({"camera_id": camera["camera_id"], "trigger": "manual"})
            cycles += 1
            if not accelerated:
                time.sleep(1 / cycles_per_second)
        current, peak = tracemalloc.get_traced_memory()
        sighting_bytes = sum(path.stat().st_size for path in app.sightings_dir.glob("*.jpg"))
        print({"cycles": cycles, "sightings": len(app.sightings), "sighting_bytes": sighting_bytes,
               "memory_current_mb": round(current / 1_048_576, 2), "memory_peak_mb": round(peak / 1_048_576, 2)})
        if peak > 128 * 1_048_576:
            raise SystemExit("memory budget exceeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seconds", type=float, default=30); parser.add_argument("--overnight", action="store_true")
    args = parser.parse_args(); run(8 * 60 * 60 if args.overnight else args.seconds, accelerated=not args.overnight)
