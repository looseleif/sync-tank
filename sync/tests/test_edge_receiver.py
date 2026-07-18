import shutil
import tempfile
import unittest
from email.message import Message

from edge_receiver import EdgeReceiverApp


class EdgeReceiverAppTests(unittest.TestCase):
    def setUp(self):
        self.storage_dir = tempfile.mkdtemp(prefix="sync-tank-edge-")
        self.app = EdgeReceiverApp(storage_dir=self.storage_dir, allowed_hub_id="tank-pi-001")

    def tearDown(self):
        shutil.rmtree(self.storage_dir)

    def test_heartbeat_stores_node_metadata(self):
        result = self.app.handle_heartbeat(
            {
                "node_id": "tank-cam-001",
                "node_type": "perimeter_camera_node",
                "hub_id": "tank-pi-001",
                "wifi_rssi": -45,
                "status": "online",
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self.app.nodes["tank-cam-001"]["wifi_rssi"], -45)
        self.assertIn("last_heartbeat_at", self.app.nodes["tank-cam-001"])

    def test_rejects_misassigned_heartbeat(self):
        with self.assertRaises(ValueError):
            self.app.handle_heartbeat({"node_id": "tank-cam-001", "hub_id": "tank-pi-002"})

    def test_raw_jpeg_upload_updates_latest_image(self):
        headers = Message()
        headers["X-Node-Id"] = "tank-cam-001"
        headers["X-Node-Type"] = "perimeter_camera_node"
        headers["X-Hub-Id"] = "tank-pi-001"
        headers["X-Wifi-Rssi"] = "-50"
        image = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"

        result = self.app.handle_image_upload(headers, image)
        latest = self.app.get_upload("tank-cam-001", "latest.jpg")

        self.assertTrue(result["ok"])
        self.assertEqual(latest, image)
        self.assertEqual(self.app.nodes["tank-cam-001"]["latest_image_url"], "/uploads/tank-cam-001/latest.jpg")
        self.assertEqual(self.app.nodes["tank-cam-001"]["wifi_rssi"], -50)

    def test_raw_jpeg_upload_accepts_query_node_id_fallback(self):
        headers = Message()
        headers["X-Hub-Id"] = "tank-pi-001"
        image = b"\xff\xd8query-node\xff\xd9"

        result = self.app.handle_image_upload(headers, image, fallback_node_id="tank-cam-002")

        self.assertTrue(result["ok"])
        self.assertEqual(self.app.get_upload("tank-cam-002", "latest.jpg"), image)

    def test_restores_latest_images_from_storage(self):
        headers = Message()
        headers["X-Node-Id"] = "tank-cam-001"
        headers["X-Hub-Id"] = "tank-pi-001"
        image = b"\xff\xd8persisted\xff\xd9"
        self.app.handle_image_upload(headers, image)

        restored = EdgeReceiverApp(storage_dir=self.storage_dir, allowed_hub_id="tank-pi-001")

        self.assertEqual(restored.get_upload("tank-cam-001", "latest.jpg"), image)
        self.assertEqual(restored.nodes["tank-cam-001"]["latest_image_url"], "/uploads/tank-cam-001/latest.jpg")

    def test_command_is_returned_once(self):
        self.app.set_command("tank-cam-001", {"command": "stream", "duration_seconds": 30})

        self.assertEqual(self.app.pop_command("tank-cam-001")["command"], "stream")
        self.assertIsNone(self.app.pop_command("tank-cam-001"))


if __name__ == "__main__":
    unittest.main()
