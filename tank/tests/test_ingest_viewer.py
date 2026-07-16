import unittest

from sync_tank.ingest import create_ingest_app


class TestIngestViewer(unittest.TestCase):
    def test_dashboard_contains_node_config_ui(self):
        _assert_dashboard_contains_node_config_ui()


def test_dashboard_contains_node_config_ui():
    _assert_dashboard_contains_node_config_ui()


def _assert_dashboard_contains_node_config_ui():
    app = create_ingest_app()
    client = app.test_client()

    response = client.get("/")
    html = response.data.decode("utf-8")

    assert response.status_code == 200
    assert "Node Inventory" in html
    assert "Camera Sources" in html
    assert "USB Baseline Capture" in html
    assert "node-config.js" in html
