import tempfile
import unittest
from pathlib import Path

from sync_tank.ingest import create_ingest_app


class TestTankLayout(unittest.TestCase):
    def test_layout_can_be_saved(self):
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
  expected_nodes:
    - tank-cam-001
  node_angles:
    tank-cam-001: 45
""",
                encoding="utf-8",
            )
            client = create_ingest_app(config).test_client()
            layout = client.get("/api/layout").json
            layout["tanks"].append({"id": "tank-two", "label": "Second Tank", "style": "cube", "internals": []})
            layout["active_tank_id"] = "tank-two"

            response = client.post("/api/layout", json=layout)

            assert response.status_code == 200
            assert response.json["layout"]["active_tank_id"] == "tank-two"
