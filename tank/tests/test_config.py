import unittest
import json
import tempfile
from pathlib import Path

import yaml

from sync_tank.config import load_config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = load_config()

        assert config.tank_id
        assert config.host["port"] == 5050
        assert "servos" in config.arm
        assert config.arm["devices"]["lighthouse-001"]["type"] == "lighthouse"
        assert config.arm["servos"]["lighthouse_pan"]["channel"] == 1
        assert config.arm["servos"]["lighthouse_tilt"]["channel"] == 0

    def test_tank_profiles_keep_rig_ownership_separate(self):
        profiles_path = Path(__file__).resolve().parents[1] / "config" / "tank_profiles.yaml"
        profiles = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))["profiles"]

        tank_one = profiles["tank1-raydar"]
        assert tank_one["role_split"] == {
            "lighthouse": True,
            "reeflex": False,
            "note": "Tank 1 owns the Raydar pan/tilt rig and Raydar camera view.",
        }
        assert tank_one["inventory"]["lighthouse_cameras"] == 1
        assert tank_one["inventory"]["reeflex_arms"] == 0

        tank_two = profiles["tank2-reeflex"]
        assert tank_two["role_split"]["lighthouse"] is False
        assert tank_two["role_split"]["reeflex"] is True
        assert tank_two["inventory"]["lighthouse_cameras"] == 0
        assert tank_two["inventory"]["reeflex_arms"] == 1

    def test_runtime_node_identity_selects_tank_two_without_changing_tracked_config(self):
        project_config = Path(__file__).resolve().parents[1] / "config"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / "sync_tank.yaml").write_text(
                "tank_id: shared\nhost:\n  port: 5050\ningest:\n  node_config_path: config/node_config.json\n",
                encoding="utf-8",
            )
            (config_dir / "tank_profiles.yaml").write_text(
                (project_config / "tank_profiles.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (config_dir / "node_config.json").write_text(
                json.dumps({"node": {"id": "tank-pi-002"}}),
                encoding="utf-8",
            )

            config = load_config(config_dir / "sync_tank.yaml")

        assert config.tank_id == "tank-2"
        assert config.raw["selected_profile"] == "tank2-reeflex"
        assert "reeflex-001" in config.arm["devices"]
        assert "lighthouse-001" not in config.arm["devices"]
        assert config.raw["autonomy"]["reeflex"]["autostart"] is True


def test_load_default_config():
    TestConfig().test_load_default_config()
