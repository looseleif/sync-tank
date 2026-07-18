import tempfile
import unittest
from pathlib import Path

from sync_tank.ingest import create_ingest_app


class TestNodeConfig(unittest.TestCase):
    def test_node_config_and_hub_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.yaml"
            config.write_text(
                f"""
tank_id: test
ingest:
  host_id: pi-test
  host_label: Test Pi
  bind: 127.0.0.1
  port: 8080
  upload_dir: {root / "uploads"}
  state_path: {root / "state.json"}
  layout_path: {root / "layout.json"}
  node_config_path: {root / "node_config.json"}
  expected_nodes:
    - tank-cam-001
""",
                encoding="utf-8",
            )
            client = create_ingest_app(config).test_client()
            node_config = client.get("/api/node-config").json
            node_config["inventory"]["robotic_arms"] = 2
            node_config["cameras"]["tank-cam-001"]["label"] = "left floater"
            node_config["cameras"]["tank-cam-001"]["node_id"] = "raspi-sync-tank-002"
            node_config["cameras"]["tank-cam-001"]["relay_node_id"] = "pi-test"
            node_config["cameras"]["tank-cam-001"]["tank_id"] = "tank-secondary"

            save = client.post("/api/node-config", json=node_config)
            payload = client.get("/api/hub-payload").json
            pc_payload = client.get("/api/pc-hub/payload", headers={"host": "PRIVATE_IP:8080"}).json

            assert save.status_code == 200
            assert payload["inventory"]["robotic_arms"] == 2
            assert payload["cameras"][0]["label"] == "left floater"
            pc_camera = pc_payload["camera_registration"]["cameras"][0]
            assert pc_payload["node"]["tank_ids"] == ["tank-secondary"]
            assert pc_camera["node_id"] == "raspi-sync-tank-002"
            assert pc_camera["relay_node_id"] == "pi-test"
            assert pc_camera["tank_id"] == "tank-secondary"
