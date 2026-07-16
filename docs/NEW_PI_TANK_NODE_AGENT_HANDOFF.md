# New Pi Tank Node Agent Handoff

Use this document to bring up a fresh Raspberry Pi as a Sync Tank tank node.

The expected hardware for this setup:

- 1 USB Lighthouse camera
- 2 USB ReefScope/endoscopic cameras
- PCA9685 servo controller at I2C address `0x40`
- Lighthouse pan/tilt servos on PCA9685 channels `0` and `1`
- Optional REEFLEX servos on channels `2`, `3`, and `4`
- ESP32 floater cameras assigned to this tank node

## Clone

```bash
cd /home/one/Projects
git clone https://github.com/looseleif/sync-tank.git sync-tank
cd /home/one/Projects/sync-tank/tank
```

If the repository already exists:

```bash
cd /home/one/Projects/sync-tank
git pull
cd tank
```

## First Run Setup

Run the interactive setup wizard:

```bash
./scripts/setup-tank.sh
```

Recommended answers for a standard tank node:

```text
Pi/node ID: tank-pi-001 or tank-pi-002
Pi/node label: TANK NODE 1 or TANK NODE 2
Tank ID this Pi serves: tank-main
Human tank label: Main Tank
ESP32 floater node IDs: tank-cam-001,tank-cam-002
Expected ReefScope USB cameras: 2
Expected Lighthouse pan/tilt cameras: 1
Expected REEFLEX arms: 1
Expected solid feeders: 1
Expected liquid feeders: 0
Expected misc feeders: 0
Tank Pi wired IP: TANK_ONE_WIRED_IP
Display/organizer Pi wired IP: SYNC_WIRED_IP
Tank dashboard URL advertised to hub: http://TANK_ONE_WIRED_IP:8080
Tank camera/control URL advertised to hub: http://TANK_ONE_WIRED_IP:5050
Install Python requirements with pip: y
Install system packages with apt: y
Enable Raspberry Pi I2C for PCA9685 servos: y
Configure eth0 static wired display link now: usually n unless directly wired to display Pi
Boot startup mode: cron
```

If I2C was enabled during setup, reboot before servo testing:

```bash
sudo reboot
```

## Start Services

After reboot:

```bash
cd /home/one/Projects/sync-tank/tank
./scripts/start-tank.sh
```

Expected services:

```text
Tank dashboard / ESP32 ingest:
  http://TANK_ONE_WIRED_IP:8080

USB camera / servo control:
  http://TANK_ONE_WIRED_IP:5050
```

## Preflight

Run:

```bash
./scripts/preflight-tank.sh
```

Confirm:

```text
config/tank_identity.yaml exists
ffmpeg exists
v4l2-ctl exists
i2cdetect exists
/dev/i2c-1 exists
/dev/video* devices exist
http://127.0.0.1:8080/api/pc-hub/payload returns 200
http://127.0.0.1:5050/api/arm returns 200
```

## USB Camera Test

Run:

```bash
./scripts/self-test-cameras.sh --repair
```

The target result for this node is:

```text
3 working USB video feeds total:
  2 ReefScope/endoscopic cameras
  1 Lighthouse camera
```

The hub/display side should eventually receive unique camera IDs such as:

```text
usb_0
usb_2
usb_4
```

Exact IDs can differ depending on Linux video enumeration. Do not reuse a camera ID for multiple feeds.

Snapshot checks:

```bash
curl -v http://127.0.0.1:5050/api/cameras/usb_0/snapshot --output /tmp/usb_0.jpg
curl -v http://127.0.0.1:5050/api/cameras/usb_2/snapshot --output /tmp/usb_2.jpg
curl -v http://127.0.0.1:5050/api/cameras/usb_4/snapshot --output /tmp/usb_4.jpg
```

MJPEG stream checks:

```bash
curl --max-time 5 -v http://127.0.0.1:5050/api/cameras/usb_0/stream --output /tmp/usb_0.mjpg
curl --max-time 5 -v http://127.0.0.1:5050/api/cameras/usb_2/stream --output /tmp/usb_2.mjpg
curl --max-time 5 -v http://127.0.0.1:5050/api/cameras/usb_4/stream --output /tmp/usb_4.mjpg
```

## Mark Camera Roles

Open the tank dashboard:

```text
http://<tank-pi-ip>:8080
```

In setup/device inventory:

- mark 2 USB feeds as ReefScope/endoscopic cameras
- mark 1 USB feed as Lighthouse camera
- keep 2 ESP32 feeds as Floater cameras
- confirm 1 REEFLEX
- confirm feeder count
- save/validate the node inventory by hand

The tank node should report active devices, not pretend old saved layouts are physical truth.

## Servo / PCA9685 Test

Confirm I2C:

```bash
ls -l /dev/i2c*
i2cdetect -y 1
```

Expected PCA9685 address:

```text
0x40
```

Check servo status:

```bash
curl -s http://127.0.0.1:5050/api/arm
```

Expected:

```text
driver: pca9685
```

If driver starts with `mock_`, do not expect physical motion. Fix I2C first.

Guarded servo tests:

```bash
./scripts/test-servos.sh 0 91
./scripts/test-servos.sh 1 91
```

Lighthouse pan/tilt test:

```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"device_id":"lighthouse-001","pan":91,"tilt":90}' \
  http://127.0.0.1:5050/api/lighthouse/pose

curl -s -H 'Content-Type: application/json' \
  -d '{"device_id":"lighthouse-001","pan":90,"tilt":91}' \
  http://127.0.0.1:5050/api/lighthouse/pose

curl -s -H 'Content-Type: application/json' \
  -d '{"device_id":"lighthouse-001","pan":90,"tilt":90}' \
  http://127.0.0.1:5050/api/lighthouse/pose
```

Default servo map:

```text
lighthouse_pan:   channel 0
lighthouse_tilt:  channel 1
reeflex_base:     channel 2
reeflex_shoulder: channel 3
reeflex_elbow:    channel 4
```

## Hub / Display Contract

The display/organizer should poll:

```text
GET http://TANK_ONE_WIRED_IP:8080/api/pc-hub/payload
GET http://TANK_ONE_WIRED_IP:8080/api/hub-payload
```

The payload should include:

- tank/node identity
- camera registration records
- USB MJPEG `stream_url`
- USB `snapshot_url`
- ESP32 floater `latest_image_url`
- setup/inventory state
- servo/control URLs

Important control URLs are in:

```text
node.control_urls
```

Expected control URLs:

```text
arm_status:
  http://TANK_ONE_WIRED_IP:5050/api/arm

lighthouse_pose:
  http://TANK_ONE_WIRED_IP:5050/api/lighthouse/pose

reeflex_pose:
  http://TANK_ONE_WIRED_IP:5050/api/reeflex/pose

reeflex_stop:
  http://TANK_ONE_WIRED_IP:5050/api/arm/stop
```

The display/organizer should render the Lighthouse camera feed and provide pan/tilt controls that call `lighthouse_pose`.

## Boot Recovery

The setup wizard should install cron boot by default:

```text
@reboot /home/one/Projects/sync-tank/tank/scripts/start-tank.sh >> /home/one/Projects/sync-tank/tank/logs/boot.log 2>&1
```

Confirm:

```bash
crontab -l
```

After power cycle:

```bash
cd /home/one/Projects/sync-tank/tank
./scripts/preflight-tank.sh
```

## If Something Fails

If no servos move:

```text
Check /dev/i2c-1
Check i2cdetect -y 1 shows 0x40
Check external servo power
Check shared ground between Pi/PCA9685/servo power
Check servo plug orientation
Check channel 0 and 1 wiring
```

If USB feeds do not show:

```text
Run ./scripts/self-test-cameras.sh --repair
Check ls /dev/video*
Check ffmpeg is installed
Check v4l2-ctl is installed
Check snapshots on 5050
Check MJPEG stream URLs on 5050
```

If the display hub shows only one camera:

```text
Check /api/pc-hub/payload
Confirm every camera has a unique camera_id
Confirm every USB camera has stream_url and snapshot_url
Confirm the display hub is not deduplicating usb_0/usb_2/usb_4
```
