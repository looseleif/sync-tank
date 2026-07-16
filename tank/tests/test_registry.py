import unittest

from sync_tank.cameras.registry import CameraRegistry


class TestRegistry(unittest.TestCase):
    def test_registry_upsert_round_trip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = CameraRegistry(Path(temp_dir) / "cameras.json")

            registry.upsert({"id": "usb_0", "name": "USB", "source_type": "usb"})

            camera = registry.get("usb_0")
            assert camera["name"] == "USB"
            assert camera["last_seen"]


def test_registry_upsert_round_trip(tmp_path):
    registry = CameraRegistry(tmp_path / "cameras.json")
    registry.upsert({"id": "usb_0", "name": "USB", "source_type": "usb"})
    camera = registry.get("usb_0")
    assert camera["name"] == "USB"
    assert camera["last_seen"]
