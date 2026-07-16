import unittest

from sync_tank.uplink.client import HubClient


class TestUplink(unittest.TestCase):
    def test_disabled_hub_does_not_send(self):
        client = HubClient("tank", {"enabled": False})

        result = client.send_frame({"id": "cam_1"}, b"fake")

        assert result["state"] == "disabled"
        assert result["ok"] is False


def test_disabled_hub_does_not_send():
    client = HubClient("tank", {"enabled": False})
    result = client.send_frame({"id": "cam_1"}, b"fake")
    assert result["state"] == "disabled"
    assert result["ok"] is False


def test_disabled_pc_contract_methods_do_not_send():
    client = HubClient("tank", {"enabled": False})

    assert client.register_node({"node_id": "tank-pi-001"}) == {"state": "disabled", "ok": False}
    assert client.send_heartbeat("tank-pi-001") == {"state": "disabled", "ok": False}
    assert client.register_cameras("tank-pi-001", []) == {"state": "disabled", "ok": False}
