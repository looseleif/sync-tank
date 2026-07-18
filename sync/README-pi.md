# Sync Tank Pi setup

## Quick start

```bash
cd ~/sync-tank
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python tank_manager.py --host 0.0.0.0 --port 8765
```

## Systemd service

```bash
sudo cp /home/zero/sync-tank/tank-manager.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tank-manager.service
```

## Kiosk mode

```bash
chromium-browser --kiosk http://127.0.0.1:8765/
```

The hosted page is a full-screen display dashboard. It refreshes the LAN node and camera layout from `/api/layout`, shows registered camera feeds, and updates uploaded JPEG snapshots without reloading the browser.

## ESP32 camera registration

## Pi Zero organizer mode

In organizer mode, the Pi Zero hosts the display/intake server on port `8765` and receives camera inventory from the real edge node.

Pi Zero organizer URL:

```text
http://PI_ZERO_IP:8765
```

Accepted organizer endpoints:

```text
POST /api/cameras/register
POST /api/nodes/heartbeat
POST /api/cameras/frame
GET  /api/layout
```

Preferred data flow:

```text
Edge node TANK_ONE_WIRED_IP -> Pi Zero organizer 8765 -> local display
```

The Pi Zero accepts pushed camera inventory, and can also poll the edge node:

```bash
python scripts/edge_organizer_agent.py \
  --organizer-base http://127.0.0.1:8765 \
  --edge-base http://TANK_ONE_WIRED_IP:8080 \
  --seed-when-unreachable
```

The organizer prefers URLs, not base64 frame pushes. Edge node image/video URLs remain the source of truth:

```text
http://TANK_ONE_WIRED_IP:8080/uploads/tank-cam-001/latest.jpg
http://TANK_ONE_WIRED_IP:8080/uploads/tank-cam-002/latest.jpg
http://TANK_ONE_WIRED_IP:5050/api/cameras/usb_0/stream
http://TANK_ONE_WIRED_IP:5050/api/cameras/usb_2/stream
```

Low-rate base64 still-frame pushes are accepted at `/api/cameras/frame`, but high-FPS video should stay on the edge node and be exposed by URL.

The Pi Zero is the display/organizer dashboard, not the VLM/3D processing machine.

## Direct wired Pi-to-Pi link

For real-time local transmission over an Ethernet cable, put the display Pi and edge Pi on a tiny static subnet.

Display Pi:

```bash
cd /home/zero/sync-tank
./scripts/configure_wired_link.sh display
```

Edge Pi:

```bash
# Copy scripts/configure_wired_link.sh to the edge Pi first if needed.
./scripts/configure_wired_link.sh edge
```

Default wired addresses:

```text
Display Pi: SYNC_WIRED_IP
Edge Pi:    TANK_ONE_WIRED_IP
```

Expected wired services:

```text
Display dashboard: http://SYNC_WIRED_IP:8765
Edge inventory:    http://TANK_ONE_WIRED_IP:8080/api/pc-hub/payload
USB stream 0:      http://TANK_ONE_WIRED_IP:5050/api/cameras/usb_0/stream
USB stream 2:      http://TANK_ONE_WIRED_IP:5050/api/cameras/usb_2/stream
```

Start the display side against the wired edge node:

```bash
cd /home/zero/sync-tank
./scripts/start_wired_display.sh
./scripts/launch_display_app.sh
```

## Deep-link relay to main PC

The Pi display node can relay its organized camera inventory to the main PC backend without forwarding raw video through the Pi. This keeps the Pi light and lets the main PC pull the edge-node URLs directly.

Topology:

```text
ESP32 Floaters -> tank Pi private AP -> tank Pi -> PoE Ethernet -> Sync/display node
```

Roles:

```text
Main PC backend
  Deep-link destination, long-term dashboard, VLM/3D/simulator, storage, planning.

Display node
  Field monitor and organizer. Ingests edge-node camera metadata and shows live feeds.

Edge nodes
  Camera/servo collectors. Receive ESP32 images, expose USB streams, drive local hardware.

ESP32 nodes
  Low-power Floater camera endpoints attached only to their tank Pi's private Wi-Fi AP.
```

The tank Pi does not need internet access for this path. Its `wlan0` hosts the Floater AP, while its wired PoE connection is the upstream data path to Sync. Floaters upload only to the tank Pi; Sync reads their inventory and latest-image URLs over Ethernet.

Run the relay from the Pi display node:

```bash
python scripts/deep_link_agent.py \
  --organizer-base http://127.0.0.1:8765 \
  --deep-link-base http://MAIN_PC_IP:8765 \
  --display-base http://PI_DISPLAY_IP:8765
```

The relay sends:

```text
POST /api/nodes/register
POST /api/cameras/register
POST /api/nodes/heartbeat
```

It also attempts `POST /api/deep-link/register` with the full topology payload if the main PC implements that endpoint.

Current relay mode is URL-reference mode:

```text
relays_stream_urls: true
relays_raw_video: false
```

Use raw video/frame relay only if the main PC cannot reach the edge-node URLs directly.

## Observer motion and classifier handoff

The display Pi runs a lightweight observer for USB and ESP32 feeds. It samples snapshot URLs, detects meaningful frame changes, saves a short burst under `/observer_events/...`, and registers an observation locally.

To also send those motion events to the main PC for animal/person/object classification, start the display stack with:

```bash
SYNC_TANK_DEEP_LINK_BASE=http://MAIN_PC_IP:8765 \
SYNC_TANK_DISPLAY_BASE=http://PI_DISPLAY_IP:8765 \
./scripts/start_wired_display.sh
```

Or run just the observer:

```bash
python scripts/feed_observer_agent.py \
  --organizer-base http://127.0.0.1:8765 \
  --storage-dir /home/zero/sync-tank/storage \
  --deep-link-base http://MAIN_PC_IP:8765 \
  --display-base http://PI_DISPLAY_IP:8765
```

Observer event posts are URL-reference payloads. The main PC should accept one of:

```text
POST /api/observations/register
POST /api/classifier/observations/register
POST /api/sync-tank/observations/register
```

The payload includes `camera_id`, `tank_id`, `event_type: motion`, `motion_score`, absolute `frame_urls`, source camera metadata, and a `classifier_request` asking the home server to identify the visible organism or motion source. The Pi does not send high-FPS video frames as JSON.

## Three-Pi fleet agent

The display node can act as the operator machine for all three wired Pis:

```text
zero      display/index node  SYNC_WIRED_IP
tank-one  tank node           TANK_ONE_WIRED_IP  ssh one@TANK_ONE_WIRED_IP
tank-two  tank node           TANK_TWO_WIRED_IP  ssh two@TANK_TWO_WIRED_IP
```

Use the fleet agent from `zero`:

```bash
cd /home/zero/sync-tank
./scripts/fleet_agent.py status
./scripts/fleet_agent.py discover
./scripts/fleet_agent.py http
```

When SSH keys are not installed yet, run without `--batch` from a real terminal so SSH can ask for the password:

```bash
./scripts/bootstrap_fleet_ssh.sh
./scripts/fleet_agent.py --node tank-one discover
./scripts/fleet_agent.py --node tank-two status
./scripts/fleet_agent.py --node tank-one tree --max-depth 3
./scripts/fleet_agent.py --node tank-one cat /home/one/sync-tank/README-pi.md
./scripts/fleet_agent.py --node tank-one ls --repo-relative scripts
./scripts/fleet_agent.py --node tank-one grep "servo" --include "*.py"
./scripts/fleet_agent.py --node tank-one repo-cat scripts/tank_hub_agent.py --lines 1,180p
```

For automated checks that should fail instead of prompting:

```bash
./scripts/fleet_agent.py --batch status
```

To pull a lightweight copy of a node repo onto the display node for local analysis:

```bash
./scripts/fleet_agent.py --node tank-one snapshot
```

Snapshots are saved under `storage/fleet_snapshots/`.

## Pi edge receiver mode

The Pi edge receiver accepts ESP32 heartbeats and raw JPEG uploads on port `8080`.

```bash
python edge_receiver.py --host 0.0.0.0 --port 8080
```

ESP32 Floaters should use their owning tank Pi's private AP address (`TANK_ONE_AP_IP` for Tank One or `TANK_TWO_AP_IP` for Tank Two):

```text
POST http://TANK_AP_IP:8080/api/node/heartbeat
POST http://TANK_AP_IP:8080/api/images/upload
GET  http://TANK_AP_IP:8080/api/node/tank-cam-001/command
```

The hub agent registers this edge node's browser-readable URLs with the display/PC hub.

```bash
HUB_BASE=http://PC_OR_DISPLAY_HUB_IP:8765 PUBLIC_BASE=http://PI_LAN_IP python scripts/tank_hub_agent.py
```

If the Pi is temporarily acting as its own hub:

```bash
HUB_BASE=http://127.0.0.1:8765 PUBLIC_BASE=http://PI_LAN_IP python scripts/tank_hub_agent.py
```

Do not make ESP32 cameras upload directly to the PC hub. ESP32 devices upload to the edge receiver over the private tank AP, and the edge receiver/agent advertises `/uploads/<camera_id>/latest.jpg` to the hub over the wired link.

## Validation

```bash
curl http://127.0.0.1:8765/api/layout
curl http://127.0.0.1:8765/api/health
```

## Safe servo diagnostic

For an initial PCA9685 servo test, run a low-amplitude verification loop that moves every channel gently between 80 and 100 degrees around the 90-degree center.

```bash
cd ~/sync-tank
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install adafruit-circuitpython-pca9685
python servo_diagnostics.py --channel-count 16 --pause-seconds 0.25
```

The routine will:
- command all channels to 90 degrees first
- move them to 100 degrees
- move them to 80 degrees
- return them to 90 degrees

This keeps current draw and mechanical strain low while confirming that the I2C bus and each servo channel are responding.
