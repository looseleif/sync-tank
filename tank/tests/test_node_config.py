import tempfile
import unittest
from pathlib import Path
import shutil

from sync_tank.ingest import _is_placeholder_url, _reachable_service_url, create_ingest_app


class TestNodeConfig(unittest.TestCase):
    def test_tank_profiles_advertise_only_their_owned_rig_controls(self):
        source_profiles = Path(__file__).resolve().parents[1] / "config" / "tank_profiles.yaml"
        expectations = {
            "tank1-raydar": ("lighthouse_survey_start", "reeflex_idle_start", "tank-1"),
            "tank2-reeflex": ("reeflex_idle_start", "lighthouse_survey_start", "tank-2"),
        }
        for profile, (present, absent, tank_id) in expectations.items():
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                config_dir = root / "config"
                config_dir.mkdir()
                shutil.copyfile(source_profiles, config_dir / "tank_profiles.yaml")
                (config_dir / "node_role").write_text(profile + "\n", encoding="utf-8")
                config = config_dir / "sync_tank.yaml"
                config.write_text(
                    f"""
host:
  port: 5050
ingest:
  state_path: {root / 'state.json'}
  layout_path: {root / 'layout.json'}
  upload_dir: {root / 'uploads'}
  node_config_path: {config_dir / 'node_config.json'}
""",
                    encoding="utf-8",
                )

                payload = create_ingest_app(config).test_client().get(
                    "/api/pc-hub/payload", headers={"host": "tank-node.test:8080"}
                ).json

                self.assertIn(present, payload["node"]["control_urls"])
                self.assertNotIn(absent, payload["node"]["control_urls"])
                self.assertEqual(payload["node"]["tank_ids"], [tank_id])

    def test_checked_in_identity_migrates_stale_rig_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            node_config_path = config_dir / "node_config.json"
            node_config_path.write_text(
                '{"inventory":{"robotic_arms":1,"lighthouses":0},"cameras":{"usb_4":{"camera_type":"reeflex_cam","label":"Reeflex Camera #1","source_type":"usb_camera"}},"profile":{"id":"tank1-reeflex"}}',
                encoding="utf-8",
            )
            (config_dir / "tank_identity.yaml").write_text(
                """
tank:
  id: tank-1
inventory:
  lighthouse_cameras: 1
  reeflex_arms: 0
profile:
  id: tank1-raydar
  role_split:
    lighthouse: true
    reeflex: false
""",
                encoding="utf-8",
            )
            config = config_dir / "sync_tank.yaml"
            config.write_text(
                f"""
tank_id: tank-1
ingest:
  node_config_path: {node_config_path}
  state_path: {root / 'state.json'}
  layout_path: {root / 'layout.json'}
  upload_dir: {root / 'uploads'}
""",
                encoding="utf-8",
            )

            client = create_ingest_app(config).test_client()
            migrated = client.get("/api/node-config").json

            assert migrated["profile"]["id"] == "tank1-raydar"
            assert migrated["inventory"]["lighthouses"] == 1
            assert migrated["inventory"]["robotic_arms"] == 0
            assert migrated["cameras"]["usb_4"]["camera_type"] == "lighthouse_cam"
            assert migrated["cameras"]["usb_4"]["label"] == "Raydar Camera #1"

    def test_placeholder_urls_fall_back_to_the_request_host(self):
        assert _is_placeholder_url("http://TANK_WIRED_HOST:8080")
        assert not _is_placeholder_url("http://tank-node.local:8080")
        assert _reachable_service_url(
            "http://TANK_WIRED_HOST:5050",
            "http://tank-node.local:8080",
            5050,
        ) == "http://tank-node.local:5050"

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
            pc_camera = pc_payload["relayed_camera_registration"]["cameras"][0]
            assert pc_payload["node"]["tank_ids"] == ["tank-secondary"]
            assert pc_camera["node_id"] == "raspi-sync-tank-002"
            assert pc_camera["relay_node_id"] == "pi-test"
            assert pc_camera["tank_id"] == "tank-secondary"
