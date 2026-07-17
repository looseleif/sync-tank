# Tank2 Tank Node Handoff

Use this handoff when another agent is configuring a second Raspberry Pi tank node for Sync Tank.

## Target Role

Tank2 should run the same tank-node software as tank1, but with tank2-specific identity and ESP32 assignments.

Recommended defaults:

```text
node_id: tank-pi-002
label: TANK NODE 2
tank_id: tank-main
expected ESP32 floaters:
  - tank-cam-003
  - tank-cam-004
wired edge IP: TANK_ONE_WIRED_IP
display/organizer IP: SYNC_WIRED_IP
```

Expected local inventory for a standard tank node:

```text
2 ReefScope USB cameras
0 Lighthouse USB cameras
1 REEFLEX servo/robot system
1 solid feeder
2 ESP32 floater cameras
```

Tank 1 should cover Lighthouse. Tank 2 should cover REEFLEX. Both tanks should send the same payload shape to the sync/display node, but with unique tank/node/camera IDs.

## Install

From the cloned repo:

```bash
cd tank
./scripts/setup-tank.sh
```

For non-interactive setup:

```bash
cd tank
./scripts/install-tank.sh \
  --node-id tank-pi-002 \
  --label "TANK NODE 2" \
  --profile tank2-reeflex \
  --tank-id tank-2 \
  --tank-label "Tank 2" \
  --expected-nodes tank-cam-003,tank-cam-004 \
  --reefscope-count 2 \
  --lighthouse-count 0 \
  --reeflex-count 1 \
  --solid-feeders 1 \
  --wired-edge-ip TANK_ONE_WIRED_IP \
  --display-pi-ip SYNC_WIRED_IP \
  --install-deps \
  --install-boot cron
```

This writes the Pi identity file:

```text
tank/config/tank_identity.yaml
```

For tank2 it should declare:

```yaml
tank:
  id: tank-2
node:
  id: tank-pi-002
  label: TANK NODE 2
esp32:
  expected_nodes:
    - tank-cam-003
    - tank-cam-004
inventory:
  reefscope_cameras: 2
  lighthouse_cameras: 0
  reeflex_arms: 1
  feeders:
    solid: 1
```

Optional dependency install:

```bash
./scripts/install-tank.sh --install-deps
```

## Start

```bash
cd tank
./scripts/start-tank.sh
```

Services:

```text
ESP32 ingest / inventory:
  http://TANK_ONE_WIRED_IP:8080

USB camera / servo control:
  http://TANK_ONE_WIRED_IP:5050
```

## Wired Link

Configure the edge Pi side:

```bash
cd tank
sudo ./scripts/configure_wired_link.sh edge
```

Expected network:

```text
Display Pi eth0: SYNC_WIRED_IP/24
Edge Pi eth0:    TANK_ONE_WIRED_IP/24
```

## Hub Payload

The display/organizer should poll:

```text
GET http://TANK_ONE_WIRED_IP:8080/api/pc-hub/payload
GET http://TANK_ONE_WIRED_IP:8080/api/hub-payload
```

The payload includes:

- node identity
- owned camera inventory
- USB MJPEG stream URLs
- ESP32 latest image URLs
- device inventory
- setup validation state
- control URLs

Control URLs appear at:

```text
node.control_urls
```

Important control URLs:

```text
arm_status:       http://TANK_ONE_WIRED_IP:5050/api/arm
lighthouse_pose: http://TANK_ONE_WIRED_IP:5050/api/lighthouse/pose
reeflex_pose:    http://TANK_ONE_WIRED_IP:5050/api/reeflex/pose
reeflex_stop:    http://TANK_ONE_WIRED_IP:5050/api/arm/stop
servo_channel:   http://TANK_ONE_WIRED_IP:5050/api/servo/channel
```

## Lighthouse Pan/Tilt

The Lighthouse uses the PCA9685 servo controller:

```text
lighthouse_pan:  channel 0
lighthouse_tilt: channel 1
```

Observer command:

```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"device_id":"lighthouse-001","pan":90,"tilt":90}' \
  http://TANK_ONE_WIRED_IP:5050/api/lighthouse/pose
```

## REEFLEX

The REEFLEX 3-servo mapping is:

```text
reeflex_base:     channel 2
reeflex_shoulder: channel 3
reeflex_elbow:    channel 4
```

Observer command:

```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"device_id":"reeflex-001","base":90,"shoulder":90,"elbow":90}' \
  http://TANK_ONE_WIRED_IP:5050/api/reeflex/pose
```

## Validation

Run full preflight:

```bash
./scripts/preflight-tank.sh
```

Run camera self-test:

```bash
./scripts/self-test-cameras.sh --repair
```

Run guarded servo test:

```bash
./scripts/test-servos.sh 0 91
./scripts/test-servos.sh 1 91
```

Confirm PCA9685:

```bash
ls -l /dev/i2c*
i2cdetect -y 1
```

Expected PCA9685 address:

```text
0x40
```
