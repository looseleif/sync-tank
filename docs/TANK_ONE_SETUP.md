# Tank One Setup Notes

This note captures the Tank One configuration used during the July 2026 wired/offline setup work.

Tank One is a Raspberry Pi tank node that owns local USB camera feeds, the REEFLEX servo arm, and the Tank 1 ESP32 floater cameras. The display/main node should reach Tank One over wired Ethernet. The ESP32 floater cameras may reach Tank One over a Pi-hosted Wi-Fi access point.

## Network Roles

Tank One uses two separate local paths:

- Wired Ethernet for the display/main node.
- Wi-Fi access point mode for ESP32 floater camera uploads.

The display/main node should not depend on Tank One's normal Wi-Fi or internet access.

```text
Display/main node -> Tank One:
  SYNC_WIRED_IP -> TANK_ONE_WIRED_IP over eth0

ESP32 floaters -> Tank One:
  TANK_ONE_AP_CLIENT -> TANK_ONE_AP_IP over wlan0 AP
```

## Wired Display Link

Current wired topology:

```text
Display/main node: SYNC_WIRED_IP/24
Tank One eth0:     TANK_ONE_WIRED_IP/24
Tank Two eth0:     TANK_TWO_WIRED_IP/24
```

Tank One NetworkManager profile:

```text
Profile: sync-tank-display-link
Interface: eth0
IPv4 method: manual
IPv4 address: TANK_ONE_WIRED_IP/24
Gateway: none
Never default route: yes
Autoconnect: yes
IPv6: disabled
```

Tank One must expose services on all interfaces:

```text
8080 bind: 0.0.0.0
5050 bind: 0.0.0.0
```

The display node polls Tank One at:

```text
Payload/status:
  http://TANK_ONE_WIRED_IP:8080/api/pc-hub/payload

Fallback payload:
  http://TANK_ONE_WIRED_IP:8080/api/hub-payload

REEFLEX status:
  http://TANK_ONE_WIRED_IP:5050/api/arm

REEFLEX pose:
  http://TANK_ONE_WIRED_IP:5050/api/reeflex/pose

REEFLEX idle controls:
  http://TANK_ONE_WIRED_IP:5050/api/reeflex/idle
  http://TANK_ONE_WIRED_IP:5050/api/reeflex/idle/start
  http://TANK_ONE_WIRED_IP:5050/api/reeflex/idle/stop
```

USB camera URLs are advertised in `/api/pc-hub/payload` and should use `TANK_ONE_WIRED_IP`, not the Wi-Fi address.

## Tank One AP

Tank One can run a local access point for its ESP32-S3 Sense floater cameras.

NetworkManager AP profile:

```text
Profile: esp32-ap
SSID: TANK_ONE_AP_SSID
Password: TANK_AP_PASSWORD
Interface: wlan0
Band: 2.4 GHz
IPv4 method: shared
AP address: TANK_ONE_AP_IP/24
Autoconnect: yes
```

Normal Wi-Fi profile:

```text
Profile: MAINTENANCE_WIFI_PROFILE
SSID: MAINTENANCE_WIFI_SSID
Autoconnect: no
```

Use normal Wi-Fi only when internet access is needed for development. The final tank path should not require it.

Turn on Tank One AP:

```bash
sudo nmcli connection down MAINTENANCE_WIFI_PROFILE
sudo nmcli connection up esp32-ap
```

Restore normal Wi-Fi:

```bash
sudo nmcli connection down esp32-ap
sudo nmcli connection up MAINTENANCE_WIFI_PROFILE ifname wlan0
```

Check active networking:

```bash
nmcli connection show --active
nmcli -f GENERAL.DEVICE,GENERAL.STATE,IP4.ADDRESS,IP4.GATEWAY,IP4.ROUTE device show eth0
nmcli -f GENERAL.DEVICE,GENERAL.STATE,IP4.ADDRESS,IP4.GATEWAY,IP4.ROUTE device show wlan0
```

Check ESP32 AP clients:

```bash
ip neigh show dev wlan0
```

## ESP32 Floater Ownership

Tank One owns:

```text
tank-cam-001
tank-cam-002
```

Tank One rejects:

```text
tank-cam-003
tank-cam-004
```

Tank Two owns `tank-cam-003` and `tank-cam-004`.

Tank One AP firmware targets:

```text
SSID: TANK_ONE_AP_SSID
Password: TANK_AP_PASSWORD
Hub/AP IP: TANK_ONE_AP_IP
```

For `tank-cam-001`:

```text
GET  http://TANK_ONE_AP_IP:8080/api/node/tank-cam-001/command
POST http://TANK_ONE_AP_IP:8080/api/node/heartbeat
POST http://TANK_ONE_AP_IP:8080/api/images/upload
```

For `tank-cam-002`:

```text
GET  http://TANK_ONE_AP_IP:8080/api/node/tank-cam-002/command
POST http://TANK_ONE_AP_IP:8080/api/node/heartbeat
POST http://TANK_ONE_AP_IP:8080/api/images/upload
```

JPEG upload requirements:

```text
Method: POST
Path: /api/images/upload
Content-Type: image/jpeg
Body: raw JPEG bytes
Headers:
  X-Node-Id: tank-cam-001 or tank-cam-002
  X-Hub-Id: tank-pi-001
```

Expected AP-mode logs:

```text
TANK_ONE_AP_CLIENT GET /api/node/tank-cam-001/command 204
TANK_ONE_AP_CLIENT POST /api/node/heartbeat 200
TANK_ONE_AP_CLIENT POST /api/images/upload 200
```

## REEFLEX

Tank One is REEFLEX-only. It does not own a Lighthouse.

REEFLEX uses PCA9685 channels:

```text
reeflex_base     -> channel 0
reeflex_shoulder -> channel 1
reeflex_elbow    -> channel 2
```

The current REEFLEX camera is advertised as:

```text
camera_id: usb_4
camera_type: reeflex_cam
label: REEFLEX Camera #1
```

Manual pose:

```bash
curl -X POST http://TANK_ONE_WIRED_IP:5050/api/reeflex/pose \
  -H 'Content-Type: application/json' \
  -d '{"base":90,"shoulder":90,"elbow":90}'
```

Idle exploration mode:

```bash
curl -X POST http://TANK_ONE_WIRED_IP:5050/api/reeflex/idle/start \
  -H 'Content-Type: application/json' \
  -d '{"amplitude":6,"period_seconds":9,"step_seconds":0.35}'
```

Manual pose or raw servo control automatically stops idle mode. This lets the observer UI run a gentle small-arc scan until the operator touches the controls.

Stop idle:

```bash
curl -X POST http://TANK_ONE_WIRED_IP:5050/api/reeflex/idle/stop
```

Check status:

```bash
curl http://TANK_ONE_WIRED_IP:5050/api/arm
curl http://TANK_ONE_WIRED_IP:5050/api/reeflex/idle
```

## USB Cameras

Tank One advertises USB cameras to the display node through `/api/pc-hub/payload`.

Current intended inventory:

```text
usb_0 -> ReefScope Camera #1
usb_2 -> ReefScope Camera #2
usb_4 -> REEFLEX Camera #1
```

The display node should render USB cameras from `stream_url` as MJPEG. It may fall back to `snapshot_url` if a stream stalls.

The tank node stream code retries ffmpeg when a stream drops, so a camera hiccup should not permanently kill the browser feed.

## Services

Tank One uses two systemd services:

```text
sync-tank.service
  Runs: python3 -m sync_tank.server
  Port: 5050
  Purpose: REEFLEX/servo control and legacy camera API

sync-tank-ingest.service
  Runs: python3 -m sync_tank.ingest
  Port: 8080
  Purpose: ESP32 ingest, USB feed URLs, display payload
```

Both should be enabled:

```bash
systemctl is-enabled sync-tank.service sync-tank-ingest.service
```

Check service health:

```bash
sudo systemctl status sync-tank.service sync-tank-ingest.service --no-pager
```

Watch ingest:

```bash
sudo journalctl -u sync-tank-ingest.service -f
```

## Offline Validation

From the display/main node:

```bash
curl http://TANK_ONE_WIRED_IP:8080/api/pc-hub/payload
curl http://TANK_ONE_WIRED_IP:8080/api/hub-payload
curl http://TANK_ONE_WIRED_IP:5050/api/arm
curl http://TANK_ONE_WIRED_IP:8080/api/usb/usb_4/snapshot --output reeflex.jpg
```

If these work while Tank One Wi-Fi is off, the wired/offline path is healthy.

Tank One should remain visible on the display node as long as:

- eth0 has `TANK_ONE_WIRED_IP/24`.
- `sync-tank.service` is running.
- `sync-tank-ingest.service` is running.
- The display node polls every 5 seconds with a stale timeout longer than that, currently 20 seconds.

## Important Separation

Use the right IP for the right job:

```text
ESP32 cameras:
  use TANK_ONE_AP_IP

Display/main node:
  use TANK_ONE_WIRED_IP

Do not make the display node use TANK_ONE_AP_IP.
Do not make ESP32 cameras depend on TANK_ONE_WIRED_IP.
Do not depend on REDACTED_PRIVATE_IP for production display traffic.
```

`REDACTED_PRIVATE_IP` was the normal Wi-Fi address during development. It disappears when `auto` is disabled, so it must not be used by the main display path.
