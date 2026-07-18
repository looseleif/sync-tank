![Sync Tank — Open-Source Aquaristics](images/sync-tank-banner.png)

# Sync Tank

Sync Tank is an open-source camera, simulation, and wildlife-observation platform for aquariums. Raspberry Pi tank nodes collect camera and device data, while a separate Sync controller combines multiple tanks into one portrait dashboard called **SEE SEA TV**.

The project is designed to remain useful without cloud services or connected hardware. Cameras and rigs may be absent, local analysis may be disabled, and the complete dashboard, simulator, Sightings album, and fake-node test environment will still run.

> Water and low voltage have never made this much sense.

## The project in 2026

Sync Tank returned to active development around Open Sauce 2026 with a multi-tank controller, a camera-first interface, offline simulation, local motion observation, and stricter motion-control safety.

<p align="center">
  <img src="images/readme/open-sauce-2026-chase-and-kara.jpg" alt="Chase and Kara standing outside an Open Sauce 2026 exhibit hall in San Mateo" width="560">
</p>

*Chase and Kara outside one of the main exhibit halls at Open Sauce 2026 in San Mateo, California.*

## SSTV then / SEE SEA TV now

The original SSTV experiments proved that several inexpensive cameras could provide useful views into an aquarium. SEE SEA TV turns those feeds into a portrait operations display with two-tank awareness, spatial context, camera rotation, local motion events, and captured Sightings.

### Blue interface prototype

The blue interface established the portrait layout and the first attempt at a useful digital copy of the installation. Tank boundaries, nearby hardware, cameras, viewing frustums, and newly placed interior objects were mapped into the same spatial view. It also introduced motion focus, a device manifest, and feed controls. It was an important working prototype, but the dense blue treatment competed with the camera imagery.

<table>
  <tr>
    <td width="50%"><img src="images/readme/see-sea-tv-blue-ui-overview.jpg" alt="Full portrait view of the earlier blue SEE SEA TV interface"></td>
    <td width="50%"><img src="images/readme/see-sea-tv-blue-simulator-detail.jpg" alt="Close view of the blue tank simulator and camera field-of-view guides"></td>
  </tr>
  <tr>
    <td><em>The complete blue portrait interface with simulator, feeds, device status, and motion focus.</em></td>
    <td><em>An early close view of the spatial simulator and camera geometry.</em></td>
  </tr>
</table>

### Current black interface

The current interface uses a restrained black presentation with a clearer hierarchy. The simulator remains large enough to explain where each view originates, while SEE SEA TV and the dominant live camera output receive most of the display.

<p align="center">
  <img src="images/readme/see-sea-tv-dry-bench-demo.jpg" alt="In-progress SEE SEA TV dry-bench demo showing the tank simulation above a live camera feed" width="560">
</p>

*An in-progress dry-bench demo. The simulated tank and camera geometry appear above a live feed while the system is tested outside the water.*

<p align="center">
  <img src="images/readme/see-sea-tv-in-water-test-setup.jpg" alt="Two aquarium test setups with cameras and control hardware being tested in water before the display was installed" width="760">
</p>

*The in-water test setup taking shape across both tanks, with camera and motion hardware connected before the portrait display was added.*

### What changed

| Earlier project language | Current user-visible name | What it means now |
| --- | --- | --- |
| SSTV | **SEE SEA TV** | The rotating, camera-first multi-tank display |
| Lighthouse | **Raydar** | The pan-and-tilt survey camera; legacy IDs and URLs remain compatible |
| ReefScope / endoscope | **Reel / Reels** | Compact fixed aquarium camera views |
| REEFLEX | **Reeflex** | The motorized inspection rig, written as a normal name |
| Saved frame | **Sighting** | A captured observation with image, source, scores, label, and notes |
| Remote captioning experiments | **Ask the Deep** | A manual-only AI field note for an already captured Sighting |

Floater cameras remain available as spatial markers. Their still images appear only when a frame changes or a marker is opened, so they no longer cover the primary camera view.

## A digital copy of the tanks

The simulator is not just decoration. It gives the live system a shared spatial vocabulary:

- Two separately labeled tanks can be viewed together or individually.
- `FRONT`, `BACK`, `LEFT`, and `RIGHT` establish orientation.
- Every camera can be represented at its physical location with a field-of-view frustum.
- The active SEE SEA TV source is highlighted so the viewer can connect footage to its real direction.
- Block, slab, rock, pillar, arch, and mound landmarks can be placed on a normalized grid.
- Interior objects can be moved, rotated, scaled, labeled, colored, duplicated, or scattered for quick layout planning.
- Orbit, pan, drag, zoom, overview, and tank-focus controls keep the model navigable on the portrait display.

This creates a practical digital copy of each tank and its surroundings: cameras describe how the habitat is being seen, while structures and landmarks describe what is being seen and where.

## Shrimp City

Shrimp City turns the aquarium interior into a recognizable habitat rather than an empty camera box. Its structures provide cover for the residents while giving cameras, the simulator, and human observers meaningful landmarks for describing where an animal was seen.

<p align="center">
  <img src="images/readme/shrimp-in-transit.jpg" alt="Shrimp being transported in a small clear container before acclimation" width="760">
</p>

*Some of Shrimp City's residents in transit, held safely in a small container before arriving at the tank and beginning acclimation.*

<table>
  <tr>
    <td width="40%"><img src="images/readme/shrimp-city-caridina-culls-street-level.jpg" alt="Caridina culls moving between the lower structures of Shrimp City"></td>
    <td width="60%"><img src="images/readme/shrimp-city-caridina-culls-wide.jpg" alt="Wide interior view of Shrimp City with Caridina culls throughout the habitat"></td>
  </tr>
  <tr>
    <td><em>Street-level activity between the structures.</em></td>
    <td><em>A wider view of the Caridina culls exploring Shrimp City.</em></td>
  </tr>
</table>

## Hardware development

The software grows alongside ordinary aquarium care and physical prototyping. Water chemistry is checked directly, while servo hardware, controller boards, wiring, and printed parts are evaluated on the bench before they approach a live tank.

<p align="center">
  <img src="images/readme/water-testing-and-hardware-prototypes.jpg" alt="Freshwater test kit beside servo hardware, a controller board, wiring, and an untested 3D-printed dispensing concept" width="760">
</p>

*Freshwater testing and early hardware laid out on the bench. The black 3D-printed dispensing concept shown here was untested and was never installed as a supported system feature.*

### Inside Reeflex

Reeflex is a motorized inspection platform built around printed mechanical parts, servos, and a PCA9685 controller. The base uses a ring of bearings to support rotation while a geared servo provides motion. Its control board separates multi-channel servo signaling from the Sync controller's higher-level safety and survey logic.

<table>
  <tr>
    <td width="50%"><img src="images/readme/reeflex-base-bearings-and-drive.jpg" alt="Open Reeflex base showing its circular bearing track, printed gear, and drive servo"></td>
    <td width="50%"><img src="images/readme/reeflex-servo-control-board.jpg" alt="PCA9685 servo control board and wiring mounted on Reeflex"></td>
  </tr>
  <tr>
    <td><em>The bearing track, printed drive gear, and servo inside the Reeflex base.</em></td>
    <td><em>The Reeflex PCA9685 servo controller and field wiring during assembly.</em></td>
  </tr>
</table>

## What works today

### SEE SEA TV

- Rotates available MJPEG feeds every eight seconds.
- Presents one dominant feed with thumbnails, previous/next navigation, and manual pinning.
- Labels every feed with its tank, camera, state, and live status.
- Keeps camera overlays and primary controls clear of the dominant footage.
- Associates the active feed with its simulated camera and viewing direction.

### Local wildlife observation

- Performs low-resolution, low-rate OpenCV motion analysis on assigned Raydar and Reeflex feeds.
- Reports persistent interesting motion rather than claiming confirmed animal identification.
- Smooths target positions, ignores small contours, and pauses analysis while a rig moves.
- Scores short capture bursts for sharpness and target centering.
- Stores manual and eligible automatic captures in the Sightings album.
- Supports Fish, Shrimp, Snail, Coral, and Unknown labels without requiring AI.

### Raydar and Reeflex autonomy

- **Raydar** surveys a calibrated 12-point circular path, can center persistent motion with bounded movements, and returns to surveying after target loss.
- **Reeflex** uses only its advertised start, stop, idle, and pose controls for a conservative small-arc survey. It does not perform closed-loop pursuit.
- Manual controls always preempt autonomous motion.
- Missing streams or controls produce `Unavailable`, not repeated commands.
- Stale frames, rejected commands, network loss, and watchdog expiry stop further motion.
- Explicit states expose `Survey`, `Track`, `Manual`, `Unavailable`, and `STOP`.

### Sightings and Ask the Deep

A Sighting preserves the original image and its tank, camera, timestamp, trigger, focus region, scores, label, favorite state, optional crop, and optional AI field note. Automatic capture has a per-camera cooldown; the shutter remains available for unrestricted manual captures.

`✦ Ask the Deep` is deliberately manual:

1. Capture or open a Sighting.
2. Click `Ask the Deep`.
3. Confirm the exact image and the disclosure: `Sends this captured image to OpenAI for analysis`.
4. Receive a cautious field note containing visual evidence, uncertainty, a possible subject, an interesting fact, and a short researcher, pirate, or captain narration.

No detection, feed rotation, startup task, or background job sends an image to OpenAI. The API key remains in the Sync server environment, automated tests use a fake transport, and all local functions continue without a key or internet connection.

## Architecture

```text
ESP32-S3 still cameras ─┐
USB / MJPEG cameras ───┼─> Raspberry Pi tank node(s)
Raydar / Reeflex ──────┘          │
                                  │ existing payload, camera, and control URLs
                                  ▼
                         Raspberry Pi Sync controller
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              SEE SEA TV     Local vision    Sightings
               simulator      and safety       album
```

Tank nodes own local camera discovery, ESP32 JPEG ingest, servo endpoints, and machine-readable inventory. The Sync controller polls those existing interfaces, proxies camera media, maintains the combined layout, runs optional local analysis, and serves the portrait dashboard. Deployed tank URLs and internal Lighthouse identifiers do not need to change.

## Repository map

| Path | Purpose |
| --- | --- |
| [`tank/`](tank/) | Maintained Raspberry Pi tank-node service, ingest receiver, camera registry, controls, setup, and tests |
| [`sync/`](sync/) | Multi-tank controller, SEE SEA TV dashboard, simulator, vision, Sightings, fake nodes, and tests |
| [`docs/`](docs/) | Deployment and tank-specific handoff notes |
| [`archive/`](archive/) | Historical prototypes retained for reference |
| [`images/`](images/) | Project artwork, historical interface images, and README photography |

## Try it without hardware

No camera or rig is required. Start two deterministic fake tank nodes and the Sync controller in separate terminals:

```bash
python3 sync/scripts/fake_tank_node.py \
  --port 18081 --node-id fake-tank-1 --tank-id tank-1

python3 sync/scripts/fake_tank_node.py \
  --port 18082 --node-id fake-tank-2 --tank-id tank-2

python3 sync/tank_manager.py \
  --host 127.0.0.1 --port 8765 --storage-dir /tmp/sync-tank-demo
```

Open `http://127.0.0.1:8765`. The fake environment can produce empty tanks, stationary decorations, a moving fish silhouette, reflections, bubbles, camera movement, multiple targets, target loss, frozen frames, malformed media, slow streams, disconnects, latency, HTTP errors, and rejected unsafe servo commands.

For detailed fake-node registration and controls, see [`sync/README.md`](sync/README.md).

## Install a tank node

On a fresh Raspberry Pi clone:

```bash
cd tank
./scripts/setup-tank.sh
```

After setup and any required I2C reboot:

```bash
./scripts/start-tank.sh
./scripts/preflight-tank.sh
```

The tank node normally serves its camera and control API on port `5050` and ESP32 heartbeat/JPEG ingest on port `8080`. Exact identities, addresses, expected cameras, and safe servo limits belong in deployment configuration rather than application code. See [`tank/README.md`](tank/README.md) for installation and endpoint details.

## Sync-only APIs

```text
GET  /api/vision/status
POST /api/vision/raydar/start
POST /api/vision/raydar/stop
POST /api/vision/reeflex/start
POST /api/vision/reeflex/stop
POST /api/sightings/capture
GET  /api/sightings
POST /api/sightings/<id>/analyze
```

These interfaces are implemented on Sync. Existing tank-node camera, ingest, pose, stop, and payload URLs remain compatible.

## Tests

Run the Sync offline suite:

```bash
(cd sync && python3 -m unittest discover -s tests -v)
```

Run the tank-node suite:

```bash
(cd tank && PYTHONPATH="$PWD" pytest)
```

Run the deterministic soak test:

```bash
python3 sync/scripts/soak_test.py --seconds 30
python3 sync/scripts/soak_test.py --overnight
```

The test environment uses virtual servos and a completely mocked OpenAI transport. It must not move physical hardware, spend API credits, or transmit captured images.

## Hardware acceptance

Simulation is intentionally not treated as proof that a physical rig is safe. Before autonomous operation on site:

1. Verify every camera, tank assignment, stream, snapshot, and control URL independently.
2. Calibrate the Raydar center and conservative pan/tilt limits.
3. Exercise STOP before moving through survey waypoints at minimum speed.
4. Verify Reeflex start, dwell, stop, and manual preemption under supervision.
5. Compare simulated camera frustums with physical viewing directions.
6. Tune motion thresholds using real reflections, bubbles, plants, and animals.
7. Validate manual and automatic captures without enabling AI analysis.
8. Complete a supervised multi-hour soak before unattended surveying.
9. Make one deliberate Ask the Deep request with a non-sensitive test image and verify the disclosure and stored result.

## License and acknowledgments

Sync Tank is released under the [MIT License](LICENSE). The project is built in the spirit of open science, open hardware, and open curiosity.

Thanks to the robotics, maker, aquarist, and open-source communities whose tools and experiments made this work possible—and to Kara, who kicks ass.
