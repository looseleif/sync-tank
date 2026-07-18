# Sync node controller

This directory contains the Sync dashboard and connector processes that coordinate multiple tank nodes.

## Responsibilities

- Render the multi-tank dashboard and camera views.
- Poll tank edge nodes for camera inventory and health.
- Proxy snapshots and streams from tank nodes to the dashboard.
- Organize observations and device state across tanks.
- Receive ESP32 heartbeats and JPEG uploads when deployed as an edge receiver.
- Run SEE SEA TV, the two-tank simulator, local motion analysis, safe Raydar seeking, bounded Reeflex control, and the Sightings album.

## Main entry points

- `tank_manager.py`: dashboard HTTP server, layout state, and camera proxy.
- `edge_receiver.py`: ESP32 heartbeat, command, and JPEG receiver.
- `scripts/start_display_stack.sh`: starts the controller and display stack.
- `scripts/start_wired_display.sh`: configures the tank-node connector polling.
- `static/`: browser dashboard.
- `config/fleet_nodes.json`: known fleet-node configuration.
- `wildlife_system.py`: testable vision, capture scoring, rig safety, and manual field-note primitives.
- `scripts/fake_tank_node.py`: deterministic fake snapshots, MJPEG, controls, failures, and virtual servos.

See `README-pi.md` for deployment and endpoint details.

## Runtime data

Runtime state, uploaded images, generated frames, logs, caches, and virtual environments are intentionally excluded from Git. Each deployed controller creates those locally.

## Offline development

No connected camera or rig is required. Missing hardware is displayed as `Unavailable`; it is not a dashboard failure. Start two fake tank nodes and Sync in three terminals:

```bash
python3 sync/scripts/fake_tank_node.py --port 18081 --node-id fake-tank-1 --tank-id tank-1
python3 sync/scripts/fake_tank_node.py --port 18082 --node-id fake-tank-2 --tank-id tank-2
python3 sync/tank_manager.py --host 127.0.0.1 --port 8765 --storage-dir /tmp/sync-tank-demo
```

Register each fake payload with `POST /api/cameras/register`, then open `http://127.0.0.1:8765`. Each fake node exposes four feeds plus deterministic `empty`, `decorations`, `fish`, `reflections`, `bubbles`, `camera-movement`, target-loss, malformed, frozen, slow, and disconnected modes through `POST /api/fake/config`. Unsafe virtual-servo poses return HTTP 422 and every accepted pose is recorded at `GET /api/fake/state`.

Run the full offline suite and a short soak:

```bash
(cd sync && python3 -m unittest discover -s tests -v)
python3 sync/scripts/soak_test.py --seconds 30
python3 sync/scripts/soak_test.py --overnight
```

Install `sync/requirements.txt` on the Sync Pi to enable OpenCV background subtraction. Without it, the dashboard, layout, manual captures, albums, fake nodes, and rig STOP controls continue to work.

## Wildlife APIs

- `GET /api/vision/status`
- `POST /api/vision/raydar/start` and `/stop`
- `POST /api/vision/reeflex/start` and `/stop`
- `POST /api/sightings/capture`
- `GET /api/sightings`
- `POST /api/sightings/<id>/analyze`

Internal `lighthouse_*` URLs and IDs remain compatible with deployed nodes, but the dashboard calls the device Raydar. Endoscopic views are Reels, and Reeflex uses normal casing.

## Inspection camera instances

Reels, Reeflex, and Raydar are all inspection-camera instances from Sync's point of view. They differ in placement and motion:

- **Reels** are positioned by hand. A light attached at the camera end lets them inspect small, dark, or obstructed spaces.
- **Reeflex** is a motorized articulated platform with bounded direct controls. Greater autonomy is future work.
- **Raydar** is a motorized pan-and-tilt platform with automated survey and seeking behavior.

## Ask the Deep

`✦ Ask the Deep` is only available on an already-captured Sighting. The confirmation dialog shows the exact image and permanently says: `Sends this captured image to OpenAI for analysis`. Detection, rotation, startup, automatic capture, and background tasks never invoke OpenAI.

Set secrets only in the Sync server environment:

```bash
export OPENAI_API_KEY='...'
export OPENAI_VISION_MODEL='gpt-5.6-terra'  # optional; this is the default
```

Automated tests inject a fake transport and cannot spend credits or transmit images.

## Guarded on-site acceptance

1. Confirm every camera ID, tank owner, stream/snapshot URL, and control URL.
2. Verify one camera at a time before enabling eight-second rotation.
3. Calibrate Raydar center and safe pan/tilt limits; exercise STOP first.
4. Run the 12 Raydar survey waypoints slowly under supervision.
5. Verify Reeflex manual poses, limits, stop, and control preemption.
6. Compare simulator frustums to physical directions and adjust placement.
7. Tune reflections, bubbles, plant motion, and target-size thresholds with aquarium footage.
8. Test tracking at the minimum command speed before increasing responsiveness.
9. Validate manual capture, one-second centered auto-capture, and per-camera cooldown.
10. Complete a supervised multi-hour soak before unattended surveying.
11. Make one deliberate Ask the Deep request using a non-sensitive test image and verify disclosure, saved field note, and API reporting.
