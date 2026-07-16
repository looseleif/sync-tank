import unittest

from sync_tank.config import load_config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = load_config()

        assert config.tank_id
        assert config.host["port"] == 5050
        assert "servos" in config.arm


def test_load_default_config():
    TestConfig().test_load_default_config()
