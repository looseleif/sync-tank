import unittest
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


def test_load_default_config():
    TestConfig().test_load_default_config()
