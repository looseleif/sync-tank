import unittest
import tempfile
import shutil
import time

from tank_manager import TankManagerApp


class TankManagerAppTests(unittest.TestCase):
    def setUp(self):
        self.storage_dir = tempfile.mkdtemp(prefix="sync-tank-manager-")
        self.app = TankManagerApp(storage_dir=self.storage_dir)

    def tearDown(self):
        shutil.rmtree(self.storage_dir)

    def test_layout_contains_registered_nodes_and_cameras(self):
        self.app.register_node(
            {
                "node_id": "raspi-sync-tank-001",
                "node_type": "raspberry_pi_tank_node",
                "hostname": "raspberrypi",
                "label": "Tank Display Pi",
                "tank_ids": ["tank-main"],
                "lan_url": "http://pi.local:8765",
                "status": "online",
            }
        )
        self.app.register_cameras(
            [
                {
                    "camera_id": "tank-cam-001",
                    "camera_type": "floater_cam",
                    "source_type": "esp32_upload",
                    "node_id": "raspi-sync-tank-001",
                    "tank_id": "tank-main",
                    "latest_image_url": "http://pi.local:8765/uploads/tank-cam-001/latest.jpg",
                    "status": "online",
                }
            ]
        )

        layout = self.app.get_layout()
        self.assertEqual(layout["nodes"][0]["node_id"], "raspi-sync-tank-001")
        self.assertEqual(layout["cameras"][0]["camera_id"], "tank-cam-001")

    def test_frame_upload_updates_latest_snapshot(self):
        self.app.register_cameras(
            [
                {
                    "camera_id": "tank-cam-001",
                    "camera_type": "floater_cam",
                    "source_type": "esp32_upload",
                    "node_id": "raspi-sync-tank-001",
                    "tank_id": "tank-main",
                    "status": "online",
                }
            ]
        )
        payload = {
            "camera_id": "tank-cam-001",
            "content_type": "image/jpeg",
            "image_base64": "AAECAwQFBgc=",
        }
        self.app.handle_frame_upload(payload)
        snapshot = self.app.get_snapshot_bytes("tank-cam-001")
        self.assertIsNotNone(snapshot)

    def test_combined_edge_payload_registers_node_and_cameras(self):
        result = self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "node_type": "raspberry_pi_tank_node",
                "label": "EDGE NODE 1",
                "tank_ids": ["tank-main"],
                "lan_url": "http://REDACTED_PRIVATE_IP:8080",
                "camera_service_url": "http://REDACTED_PRIVATE_IP:5050",
                "status": "online",
                "cameras": [
                    {
                        "camera_id": "tank-cam-001",
                        "camera_type": "floater_cam",
                        "source_type": "esp32_upload",
                        "node_id": "tank-pi-001",
                        "tank_id": "tank-main",
                        "latest_image_url": "http://REDACTED_PRIVATE_IP:8080/uploads/tank-cam-001/latest.jpg",
                        "status": "online",
                    },
                    {
                        "camera_id": "usb_0",
                        "camera_type": "endoscope_cam",
                        "source_type": "usb_camera",
                        "node_id": "tank-pi-001",
                        "tank_id": "tank-main",
                        "snapshot_url": "http://REDACTED_PRIVATE_IP:5050/api/cameras/usb_0/snapshot",
                        "stream_url": "http://REDACTED_PRIVATE_IP:5050/api/cameras/usb_0/stream",
                        "status": "online",
                    },
                ],
            }
        )

        layout = self.app.get_layout()
        self.assertEqual(result["registered_cameras"], 2)
        self.assertEqual(layout["nodes"][0]["node_id"], "tank-pi-001")
        self.assertEqual(layout["nodes"][0]["camera_service_url"], "http://REDACTED_PRIVATE_IP:5050")
        self.assertEqual(layout["cameras"][1]["camera_id"], "usb_0")
        self.assertIn("tank-main", {tank["tank_id"] for tank in layout["tanks"]})

    def test_connected_tank_replaces_empty_default_tank(self):
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "tank_ids": ["tank-main"],
                "status": "online",
                "cameras": [
                    {
                        "camera_id": "usb_0",
                        "camera_type": "endoscope_cam",
                        "tank_id": "tank-main",
                        "status": "online",
                    }
                ],
            }
        )

        layout = self.app.get_layout()

        self.assertEqual(layout["tanks"][0]["tank_id"], "tank-main")
        self.assertEqual(layout["scene_items"], [])

    def test_layout_persists_registered_edge_payload(self):
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "status": "online",
                "cameras": [{"camera_id": "tank-cam-001", "status": "online"}],
            }
        )

        restored = TankManagerApp(storage_dir=self.storage_dir)

        self.assertEqual(restored.nodes["tank-pi-001"]["status"], "online")
        self.assertEqual(restored.cameras["tank-cam-001"]["status"], "online")

    def test_edge_payload_preserves_device_inventory(self):
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "status": "online",
                "device_inventory": {
                    "active_tank_nodes": 1,
                    "owned_device_count": 5,
                    "counts": {"scope": 2, "reeflex": 1, "lighthouse": 1, "floater": 1},
                },
                "device_catalog": [{"device_type": "scope", "label": "ReefScope", "status": "available"}],
                "setup_state": {"status": "default", "validated_by_hand": False},
                "cameras": [{"camera_id": "usb_0", "source_type": "usb_camera", "status": "online"}],
            }
        )

        node = self.app.get_layout()["nodes"][0]

        self.assertEqual(node["device_inventory"]["counts"]["scope"], 2)
        self.assertEqual(node["device_inventory"]["counts"]["reeflex"], 1)
        self.assertFalse(node["setup_state"]["validated_by_hand"])

    def test_multi_node_payloads_do_not_replace_each_other(self):
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "tank_ids": ["tank-1"],
                "status": "online",
                "cameras": [
                    {"camera_id": "tank-cam-001", "node_id": "tank-pi-001", "tank_id": "tank-1", "status": "online"},
                    {"camera_id": "tank-cam-002", "node_id": "tank-pi-001", "tank_id": "tank-1", "status": "online"},
                ],
            }
        )
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-002",
                "tank_ids": ["tank-2"],
                "status": "online",
                "cameras": [
                    {"camera_id": "tank-cam-003", "node_id": "tank-pi-002", "tank_id": "tank-2", "status": "online"},
                    {"camera_id": "tank-cam-004", "node_id": "tank-pi-002", "tank_id": "tank-2", "status": "online"},
                ],
            }
        )

        self.assertEqual(
            set(self.app.cameras),
            {"tank-cam-001", "tank-cam-002", "tank-cam-003", "tank-cam-004"},
        )

    def test_replace_payload_is_scoped_to_reporting_node(self):
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "tank_ids": ["tank-1"],
                "status": "online",
                "cameras": [
                    {"camera_id": "tank-cam-001", "node_id": "tank-pi-001", "tank_id": "tank-1", "status": "online"},
                    {"camera_id": "tank-cam-002", "node_id": "tank-pi-001", "tank_id": "tank-1", "status": "online"},
                ],
            }
        )
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-002",
                "tank_ids": ["tank-2"],
                "status": "online",
                "cameras": [
                    {"camera_id": "tank-cam-003", "node_id": "tank-pi-002", "tank_id": "tank-2", "status": "online"},
                ],
            }
        )
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "tank_ids": ["tank-1"],
                "status": "online",
                "cameras": [
                    {"camera_id": "tank-cam-001", "node_id": "tank-pi-001", "tank_id": "tank-1", "status": "online"},
                ],
            }
        )

        self.assertEqual(set(self.app.cameras), {"tank-cam-001", "tank-cam-003"})

    def test_active_nodes_payload_marks_recent_node_active(self):
        self.app.register_node(
            {
                "node_id": "tank-pi-001",
                "status": "online",
                "tank_ids": ["tank-main"],
            }
        )

        active = self.app.active_nodes_payload()

        self.assertEqual(active["active_node_ids"], ["tank-pi-001"])
        self.assertEqual(active["active_count"], 1)

    def test_stale_node_marks_node_and_cameras_offline(self):
        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "status": "online",
                "cameras": [
                    {
                        "camera_id": "usb_0",
                        "node_id": "tank-pi-001",
                        "status": "online",
                    }
                ],
            }
        )
        self.app.nodes["tank-pi-001"]["last_seen_at"] = time.time() - 999

        layout = self.app.get_layout()

        self.assertFalse(layout["nodes"][0]["active"])
        self.assertEqual(layout["nodes"][0]["status"], "offline")
        self.assertEqual(layout["cameras"][0]["status"], "offline")

    def test_edge_payload_replaces_stale_camera_inventory_for_reporting_node(self):
        self.app.register_cameras(
            [
                {"camera_id": "stale-cam", "node_id": "old-node"},
                {"camera_id": "tank-cam-001", "node_id": "tank-pi-001"},
            ]
        )

        self.app.register_camera_payload(
            {
                "node_id": "tank-pi-001",
                "cameras": [{"camera_id": "tank-cam-002", "node_id": "tank-pi-001"}],
            }
        )

        self.assertIn("stale-cam", self.app.cameras)
        self.assertNotIn("tank-cam-001", self.app.cameras)
        self.assertIn("tank-cam-002", self.app.cameras)

    def test_register_and_label_observation(self):
        observation = self.app.register_observation(
            {
                "observation_id": "obs-1",
                "camera_id": "tank-cam-001",
                "tank_id": "tank-main",
                "event_type": "motion",
                "identity_status": "needs_label",
                "frame_urls": ["/observer_events/obs-1/frame-1.jpg"],
            }
        )

        self.assertEqual(observation["identity_status"], "needs_label")
        self.assertEqual(self.app.get_layout()["observations"][0]["observation_id"], "obs-1")

        labeled = self.app.label_observation({"observation_id": "obs-1", "label": "Fish"})

        self.assertEqual(labeled["label"], "Fish")
        self.assertEqual(labeled["identity_status"], "labeled")
        self.assertIn("fish", self.app.organisms)


if __name__ == "__main__":
    unittest.main()
