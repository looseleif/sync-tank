# Pi Zero Agent Handoff: Sync Tank Camera Transfer Modes

## Goal

Restore the original wireless camera transfer path while keeping compatibility with the newer fast burst mode.

The camera nodes should be able to send JPEG data over Wi-Fi to a Raspberry Pi receiver. The receiver can be a Pi 5 edge node, a Pi Zero bridge, or another Sync Tank edge node, but the PC hub should still receive clean camera registration metadata from the owning edge node.

## Original Wireless Transfer Mode

This was the first working mode before the camera path became more local/wired.

ESP32 camera nodes:

- Wake from deep sleep.
- Connect to Wi-Fi.
- Send a heartbeat JSON to the Pi.
- Capture one JPEG.
- Upload the raw JPEG to the Pi over HTTP.
- Send another heartbeat.
- Return to deep sleep.
- Repeat on the configured wake interval.

The ESP32 devices do not need Tailscale. They should talk to the Pi using normal LAN Wi-Fi.

Example camera target:

```text
http://PI_LAN_IP:8080/api/images/upload
http://PI_LAN_IP:8080/api/node/heartbeat
```

Current known Pi 5 edge node example:

```text
http://PRIVATE_IP:8080
```

## Wi-Fi Details

Current camera Wi-Fi target:

```text
SSID: MAINTENANCE_WIFI_SSID
Password: TANK_AP_PASSWORD
```

Older tests also used:

```text
SSID: looseline
```

Use the current `auto` network unless deliberately testing the older network.

## Node IDs

Known ESP32 floater camera IDs:

```text
tank-cam-001
tank-cam-002
tank-cam-003
tank-cam-004
```

Current assignment plan:

```yaml
tank-pi-001:
  - tank-cam-001
  - tank-cam-002

tank-pi-002:
  - tank-cam-003
  - tank-cam-004
```

If a Pi Zero is acting as a relay, do not change the physical owner unless the camera actually belongs to the Pi Zero. Preserve ownership metadata and add relay metadata when needed.

Example relay metadata:

```json
{
  "camera_id": "tank-cam-003",
  "node_id": "tank-pi-002",
  "relay_node_id": "pi-zero-bridge-001",
  "tank_id": "tank-main"
}
```

## Required Pi Receiver Endpoints

The Pi-side receiver should expose these endpoints:

```text
GET  /
POST /api/images/upload
POST /api/node/heartbeat
GET  /api/node/<node_id>/command
GET  /uploads/<node_id>/<filename>.jpg
GET  /uploads/<node_id>/latest.jpg
```

The local dashboard should stay simple:

- Latest image per camera.
- Last heartbeat time.
- RSSI.
- Battery voltage when available.
- Upload status.
- Stream/burst command state.

## Heartbeat Request

Endpoint:

```text
POST /api/node/heartbeat
Content-Type: application/json
```

Payload:

```json
{
  "node_id": "tank-cam-001",
  "node_type": "perimeter_camera_node",
  "hub_id": "tank-pi-001",
  "firmware": "sync-tank-xiao-s3-camera-0.1.0",
  "uptime_ms": 12345,
  "wifi_rssi": -45,
  "free_heap": 180000,
  "battery_mv": null,
  "camera_available": true,
  "last_image_upload_status": "ok",
  "status": "online"
}
```

Receiver behavior:

- Validate `node_id`.
- Validate `hub_id` if this receiver owns only specific camera nodes.
- Store last heartbeat metadata.
- Return HTTP `200` on accepted heartbeat.
- Return HTTP `400` for malformed or misassigned nodes.

## Image Upload Request

Endpoint:

```text
POST /api/images/upload
Content-Type: image/jpeg
```

Body:

```text
raw JPEG bytes
```

Headers:

```text
X-Node-Id: tank-cam-001
X-Node-Type: perimeter_camera_node
X-Hub-Id: tank-pi-001
X-Firmware-Version: sync-tank-xiao-s3-camera-0.1.0
X-Uptime-Ms: milliseconds since wake
X-Wifi-Rssi: RSSI
X-Free-Heap: heap bytes
X-Image-Format: jpeg
X-Image-Size-Bytes: JPEG byte length
X-Capture-Timestamp-Ms: milliseconds since wake
```

Receiver behavior:

- Confirm body starts like a JPEG.
- Store under:

```text
test_uploads/<node_id>/<node_id>_<timestamp>.jpg
```

- Update each node's `latest_image`.
- Serve latest image from:

```text
/uploads/<node_id>/latest.jpg
```

Recommended success response:

```json
{
  "ok": true,
  "node_id": "tank-cam-001",
  "hub_id": "tank-pi-001",
  "latest_image": {
    "filename": "tank-cam-001_20260707T120000000000Z.jpg",
    "url": "/uploads/tank-cam-001/tank-cam-001_20260707T120000000000Z.jpg",
    "size_bytes": 12345
  }
}
```

## Command Polling

Each ESP32 wakes and asks the Pi whether it should do anything special.

Endpoint:

```text
GET /api/node/<node_id>/command
```

No command:

```text
HTTP 204 No Content
```

Capture command:

```json
{
  "command": "capture"
}
```

Stream/burst command:

```json
{
  "command": "stream",
  "duration_seconds": 30,
  "stream_seconds": 30
}
```

Receiver behavior:

- Maintain one pending command per node.
- Return the command once.
- Clear the command immediately after it is returned.
- If no command exists, return HTTP `204`.

## Fast Mode / Burst Mode

The current fast mode is not a true socket video stream.

It is a timed burst of normal JPEG uploads to the same endpoint:

```text
POST /api/images/upload
```

Expected embedded limits:

```text
STREAM_MAX_SECONDS = 60
STREAM_FRAME_INTERVAL_MS = 1000
```

That means max burst is roughly:

```text
1 JPEG/second for up to 60 seconds
```

The Pi receiver should treat rapid uploads from the same node as a burst/clip:

- Start burst tracking when a stream command is delivered.
- Add each incoming JPEG from that node to the active burst.
- Show a timeline or frame count on the dashboard.
- Allow playback of the captured burst frames after the burst completes.
- Prefer fresh frames over a large backlog.

## Pi Zero Bridge Behavior

If the Pi Zero is the receiver:

- Run the same HTTP ingest receiver on port `8080`.
- Store images by `node_id`.
- Serve dashboard at `/`.
- Serve latest image URLs.
- Optionally forward/relay metadata to the Pi 5 or PC hub.

If the Pi Zero is only a bridge:

- ESP32 uploads to the Pi Zero LAN IP.
- Pi Zero stores and/or forwards the image to the owning Pi edge node.
- PC camera metadata should still show the physical owner in `node_id`.
- Add `relay_node_id` when the Pi Zero is relaying another node's feed.

Do not make ESP32 cameras talk directly to the PC hub. The edge node layer should remain the collector/relay.

## PC Hub Registration Contract

The edge node or bridge should register cameras with the PC hub without fake placement data.

Endpoint:

```text
POST http://PC_LAN_IP:8765/api/cameras/register
```

Example:

```json
{
  "node_id": "tank-pi-001",
  "cameras": [
    {
      "camera_id": "tank-cam-001",
      "camera_type": "floater_cam",
      "source_type": "esp32_upload",
      "node_id": "tank-pi-001",
      "tank_id": "tank-main",
      "latest_image_url": "http://PRIVATE_IP:8080/uploads/tank-cam-001/latest.jpg",
      "status": "online"
    }
  ]
}
```

Do not send these fields unless there is real calibration:

```text
position
target
fov_degrees
layout_status
```

The PC hub user places the camera manually in the 3D tank map.

## Tailscale Notes

Tailscale is useful for humans viewing/administering the Pi dashboard remotely.

Tailscale is not required for ESP32 uploads.

ESP32 upload target should stay as the normal LAN Wi-Fi IP:

```text
http://PI_LAN_IP:8080
```

Phone/laptop remote viewing can use Tailscale later:

```text
http://PI_TAILSCALE_IP:8080
```

## Minimal Run Commands

From the Sync Tank host project:

```bash
cd /home/one/Projects/sync-tank/tank
./scripts/run-ingest.sh
```

Camera/control service for USB and REEFLEX side:

```bash
cd /home/one/Projects/sync-tank/tank
./scripts/run-dev.sh
```

USB self-test:

```bash
cd /home/one/Projects/sync-tank/tank
./scripts/self-test-cameras.sh --repair
```

## Quick Test From Another Machine

Send a fake heartbeat:

```bash
curl -X POST http://PI_LAN_IP:8080/api/node/heartbeat \
  -H 'Content-Type: application/json' \
  -d '{
    "node_id": "tank-cam-001",
    "node_type": "perimeter_camera_node",
    "hub_id": "tank-pi-001",
    "firmware": "manual-test",
    "uptime_ms": 1000,
    "wifi_rssi": -50,
    "free_heap": 180000,
    "battery_mv": null,
    "camera_available": true,
    "last_image_upload_status": "manual",
    "status": "online"
  }'
```

Upload a local JPEG:

```bash
curl -X POST http://PI_LAN_IP:8080/api/images/upload \
  -H 'Content-Type: image/jpeg' \
  -H 'X-Node-Id: tank-cam-001' \
  -H 'X-Node-Type: perimeter_camera_node' \
  -H 'X-Hub-Id: tank-pi-001' \
  -H 'X-Firmware-Version: manual-test' \
  -H 'X-Uptime-Ms: 2000' \
  -H 'X-Wifi-Rssi: -50' \
  -H 'X-Free-Heap: 180000' \
  -H 'X-Image-Format: jpeg' \
  -H 'X-Image-Size-Bytes: 12345' \
  -H 'X-Capture-Timestamp-Ms: 1900' \
  --data-binary @test.jpg
```

Check latest image:

```bash
curl -I http://PI_LAN_IP:8080/uploads/tank-cam-001/latest.jpg
```

## Acceptance Criteria

- Pi Zero or edge node receives heartbeat JSON.
- Pi Zero or edge node receives raw JPEG uploads.
- Latest image is visible from `/uploads/<node_id>/latest.jpg`.
- Normal deep sleep still mode works.
- Fast burst mode works through command polling.
- Burst frames are shown as a captured clip/timeline.
- PC hub registration keeps correct `node_id`, `relay_node_id`, and `tank_id`.
- ESP32 devices stay on LAN HTTP and do not need Tailscale.
