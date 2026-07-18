import unittest

from scripts.deep_link_agent import build_deep_link_payload, display_node_payload


class DeepLinkAgentTests(unittest.TestCase):
    def test_display_node_payload_identifies_local_display(self):
        payload = display_node_payload("pi-zero-display-001", "http://PRIVATE_IP:8765")

        self.assertEqual(payload["node_id"], "pi-zero-display-001")
        self.assertEqual(payload["node_type"], "sync_tank_display_node")
        self.assertEqual(payload["lan_url"], "http://PRIVATE_IP:8765")

    def test_build_deep_link_payload_preserves_edge_cameras(self):
        layout = {
            "nodes": [
                {
                    "node_id": "tank-pi-001",
                    "node_type": "raspberry_pi_tank_node",
                    "lan_url": "http://PRIVATE_IP:8080",
                    "status": "online",
                }
            ],
            "cameras": [
                {
                    "camera_id": "usb_0",
                    "camera_type": "endoscope_cam",
                    "source_type": "usb_camera",
                    "node_id": "tank-pi-001",
                    "stream_url": "http://PRIVATE_IP:5050/api/cameras/usb_0/stream",
                    "status": "online",
                }
            ],
        }

        payload = build_deep_link_payload(
            layout,
            display_node_id="pi-zero-display-001",
            display_base="http://PRIVATE_IP:8765",
        )

        self.assertEqual(payload["display_node"]["node_id"], "pi-zero-display-001")
        self.assertEqual(payload["nodes"][1]["node_id"], "tank-pi-001")
        self.assertEqual(payload["cameras"][0]["display_node_id"], "pi-zero-display-001")
        self.assertEqual(payload["cameras"][0]["deep_link_mode"], "url_reference")
        self.assertEqual(payload["cameras"][0]["stream_url"], "http://PRIVATE_IP:5050/api/cameras/usb_0/stream")


if __name__ == "__main__":
    unittest.main()
