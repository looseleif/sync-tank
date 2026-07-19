import tempfile
import unittest
from pathlib import Path

from sync_tank.ingest import create_ingest_app


class TestIngest(unittest.TestCase):
    def test_heartbeat_and_image_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _run_heartbeat_and_image_upload(Path(temp_dir))


def test_heartbeat_and_image_upload(tmp_path):
    _run_heartbeat_and_image_upload(tmp_path)


def _run_heartbeat_and_image_upload(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
tank_id: test
ingest:
  hub_id: tank-pi-001
  host_id: tank-pi-001
  bind: 127.0.0.1
  port: 8080
  upload_dir: {tmp_path / "uploads"}
  state_path: {tmp_path / "state.json"}
  layout_path: {tmp_path / "layout.json"}
  node_config_path: {tmp_path / "node_config.json"}
  camera_service_url: http://PRIVATE_IP:5050
  max_images_per_node: 5
  expected_nodes:
    - tank-cam-001
  allowed_nodes:
    - tank-cam-001
""",
        encoding="utf-8",
    )
    app = create_ingest_app(config)
    client = app.test_client()

    heartbeat = client.post(
        "/api/node/heartbeat",
        json={
            "node_id": "tank-cam-001",
            "node_type": "perimeter_camera_node",
            "hub_id": "tank-pi-001",
            "firmware": "sync-tank-xiao-s3-camera-0.1.0",
            "status": "online",
        },
    )
    assert heartbeat.status_code == 200

    upload = client.post(
        "/api/images/upload",
        data=b"\xff\xd8fake-jpeg\xff\xd9",
        headers={
            "content-type": "image/jpeg",
            "X-Node-Id": "tank-cam-001",
            "X-Node-Type": "perimeter_camera_node",
            "X-Hub-Id": "tank-pi-001",
            "X-Firmware-Version": "sync-tank-xiao-s3-camera-0.1.0",
        },
    )

    assert upload.status_code == 200
    assert upload.json["hub_id"] == "tank-pi-001"
    assert upload.json["latest_image"]["url"].startswith("/uploads/tank-cam-001/")

    latest = client.get("/uploads/tank-cam-001/latest.jpg")
    assert latest.status_code == 200

    pc_payload = client.get("/api/pc-hub/payload", headers={"host": "tank-node.test:8080"})
    assert pc_payload.status_code == 200
    assert pc_payload.json["node"]["node_id"] == "tank-pi-001"
    assert pc_payload.json["node"]["lan_url"] == "http://tank-node.test:8080"
    assert pc_payload.json["camera_registration"]["node_id"] == "tank-pi-001"
    assert pc_payload.json["camera_registration"]["cameras"][0]["latest_image_url"] == "http://tank-node.test:8080/uploads/tank-cam-001/latest.jpg"
    allowed_pc_camera_fields = {
        "id",
        "camera_id",
        "label",
        "name",
        "camera_type",
        "source_type",
        "node_id",
        "relay_node_id",
        "tank_id",
        "latest_image_url",
        "snapshot_url",
        "stream_url",
        "preferred_live_url",
        "feed_mode",
        "content_type",
        "capture_command_url",
        "supports_capture_request",
        "status",
    }
    forbidden_layout_fields = {"position", "target", "fov_degrees", "layout_status"}
    for camera in pc_payload.json["camera_registration"]["cameras"]:
        assert set(camera).issubset(allowed_pc_camera_fields)
        assert set(camera).isdisjoint(forbidden_layout_fields)
        assert camera["node_id"] == "tank-pi-001"
        assert camera["tank_id"] == "tank-main"
        if camera["source_type"] == "usb_camera":
            assert camera["snapshot_url"].startswith("http://tank-node.test:5050/api/cameras/")
            assert camera["stream_url"].startswith("http://tank-node.test:5050/api/cameras/")

    queued = client.post("/api/node/tank-cam-001/command", json={"command": "stream", "duration_seconds": 30})
    assert queued.status_code == 200
    assert queued.json["command"]["stream_seconds"] == 30

    command = client.get("/api/node/tank-cam-001/command")
    assert command.status_code == 200
    assert command.json["command"] == "stream"
    assert command.json["stream_seconds"] == 30

    empty = client.get("/api/node/tank-cam-001/command")
    assert empty.status_code == 204

    stream_request = client.post("/api/node/tank-cam-001/stream", json={"stream_seconds": 30})
    assert stream_request.status_code == 200
    assert stream_request.json["state"] == "waiting_for_node_wake"

    stream_command = client.get("/api/node/tank-cam-001/command")
    assert stream_command.status_code == 200
    assert stream_command.json == {"command": "stream", "duration_seconds": 30, "stream_seconds": 30}

    waiting_stream = client.get("/api/node/tank-cam-001/stream/status")
    assert waiting_stream.status_code == 200
    assert waiting_stream.json["session"]["status"] == "waiting_for_frames"
    assert waiting_stream.json["history"][0]["status"] == "waiting_for_frames"

    burst_frame = client.post(
        "/api/images/upload",
        data=b"\xff\xd8stream-frame\xff\xd9",
        headers={
            "content-type": "image/jpeg",
            "X-Node-Id": "tank-cam-001",
            "X-Node-Type": "perimeter_camera_node",
            "X-Hub-Id": "tank-pi-001",
            "X-Firmware-Version": "sync-tank-xiao-s3-camera-0.1.0",
        },
    )
    assert burst_frame.status_code == 200

    active_stream = client.get("/api/node/tank-cam-001/stream/status")
    assert active_stream.status_code == 200
    assert active_stream.json["session"]["status"] == "active"
    assert active_stream.json["session"]["frame_count"] == 1
    assert active_stream.json["session"]["frames"][0]["url"].startswith("/uploads/tank-cam-001/")
    assert active_stream.json["history"][0]["frame_count"] == 1

    latest_stream = client.get("/api/node/tank-cam-001/stream/latest")
    assert latest_stream.status_code == 200
    assert latest_stream.content_type == "image/jpeg"

    second_stream_request = client.post("/api/node/tank-cam-001/stream", json={"stream_seconds": 30})
    assert second_stream_request.status_code == 200
    pending_stream = client.get("/api/node/tank-cam-001/stream/status")
    assert pending_stream.status_code == 200
    assert pending_stream.json["status"] == "pending"
    assert pending_stream.json["pending_command"]["command"] == "stream"
    assert pending_stream.json["history"][0]["frame_count"] == 1

    wrong_hub = client.post(
        "/api/node/heartbeat",
        json={
            "node_id": "tank-cam-001",
            "node_type": "perimeter_camera_node",
            "hub_id": "tank-pi-002",
            "status": "online",
        },
    )
    assert wrong_hub.status_code == 400

    wrong_node = client.post(
        "/api/node/heartbeat",
        json={
            "node_id": "tank-cam-004",
            "node_type": "perimeter_camera_node",
            "hub_id": "tank-pi-001",
            "status": "online",
        },
    )
    assert wrong_node.status_code == 400
