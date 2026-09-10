![Sync Tank aquarium artwork](images/sync.jpg)

# Sync Tank

**Open-source aquaristics: connected tanks, shared perspectives, local control.**

Sync Tank brings aquarium cameras, inspection tools, habitat models, and observations into one local system. It connects what a camera sees with the tank it belongs to, where it is looking, and the equipment around it, so an aquarium can be explored and understood from more than one perspective.

Each tank has a Raspberry Pi node that collects its camera feeds and manages its devices. A shared Sync hub brings those tanks together for viewing, spatial mapping, captured observations, and inspection controls. Everyday operation stays on the local network, with footage and controls under the aquarium owner's control.

The current installation connects two independent tank nodes. That is the reference setup, with open interfaces intended to let more tanks, cameras, and tools join over time.

[Visit the aquarium shop](https://looseleif.github.io/sync-tank/) · [Build a tank in your browser](https://looseleif.github.io/sync-tank/builder.html) · [Electronics counter](https://looseleif.github.io/sync-tank/hardware.html)

## Start here

| Start with... | Where to go |
| --- | --- |
| A photo and an idea | [Tank builder](https://looseleif.github.io/sync-tank/builder.html): an empty 3D tank, photo references, movable cameras and structures, local drafts and JSON export |
| A parts list | [Electronics and wiring](docs/HARDWARE.md): Raspberry Pi, PCA9685, MG995 motors, PoE receivers and the NETGEAR switch, with product references and evidence status |
| No hardware at all | [Getting started](docs/GETTING_STARTED.md), then the [local simulated-node demo](#try-it-without-hardware) |
| A Pi and cameras | [Maintained tank-node setup](tank/README.md); identify the node role and leave motor power disconnected during initial installation |
| A model or printed part | [Design origins](docs/DESIGN_ORIGINS.md): source links, licenses, local modifications and the information still needed |

The browser builder is a photo-assisted planning sandbox, not automatic 3D
reconstruction. Front, side and top references help refine different coordinates;
one photograph cannot reveal hidden depth or physical scale. Read the
[photo modeling guide](docs/TANK_MODELING.md). Draft files do not command live hardware.

## How the pieces fit together

Sync Tank is the whole project. Its hardware and software components each serve a different part of aquarium work:

| Component | Role in Sync Tank |
| --- | --- |
| **Tank nodes and Sync hub** | Collect cameras and devices locally, preserve tank ownership, and coordinate the installation. |
| **Floaters, Reels, Reeflex, and Raydar** | Provide fixed, manually placed, and motorized camera perspectives. |
| **Spatial tank model** | Place cameras, viewing directions, and habitat landmarks in an interactive 3D map. |
| **SEE SEA TV** | Display the connected camera feeds, with navigation, rotation, and a link to their physical locations. |
| **Local observation and Sightings** | Notice persistent motion and save images with their tank, camera, time, and notes. |
| **Ask the Deep** | Add an optional AI field note to a captured Sighting when explicitly requested. |
| **Shrimp City and habitat hardware** | Give aquarium residents cover and observers recognizable places to study. |

A camera supplies a view; its tank node establishes where it belongs; the spatial model describes its position and direction. SEE SEA TV presents that view, and Sightings preserves selected moments with their context. These connections are the foundation for future observation tools and voluntary community sharing.

## The Sync Tank mission

Sync Tank exists to make aquarium work easier to connect, understand, share, and improve. Each aquarium can keep its footage, controls, and analysis on a local hub while still benefiting from a wider open-source community.

### Start with two, design for many

- Prove that separate Raspberry Pi tank nodes can report into one local Sync hub.
- Keep tank identity, camera identity, layout, and control boundaries intact as the network grows.
- Allow each node to keep working when another tank, the internet, a camera, or an optional analysis service is unavailable.
- Let viewing, mapping, inspection, and observation tools use the same tank and camera identities.
- Treat the two-tank deployment as a real reference implementation for larger personal, classroom, research, and community installations.

### More eyes on aquatic life

The long-term goal is a community of aquarists who can choose to share useful footage, Sightings, failure cases, configurations, and improvements. More people observing similar animals and systems can create earlier signals when something is unusual:

- changes in movement, appearance, hiding, feeding response, or habitat use;
- equipment failures, blocked views, stale cameras, unsafe motion, or environmental changes;
- recurring false detections and difficult visual conditions;
- software, firmware, network, and hardware vulnerabilities;
- patterns that one person or one short observation might miss.

Sync Tank should help turn those signals into understandable notifications for the people responsible for the aquarium. It is not a substitute for attentive husbandry, water testing, veterinary expertise, or human judgment; its value is helping the right person notice and investigate sooner.

### Local ownership, voluntary sharing

Community participation must not require sending every feed to a central service. Raw footage and controls stay local by default. Sharing should be explicit and selective—such as a chosen clip, a captured Sighting, an anonymized failure case, or an open-source fix. The same model lets contributors improve camera support, tests, safety limits, visual analysis, documentation, and hardware designs without needing identical tanks.

Success means that someone can add a new node or observation tool for their own aquarium, keep control of it locally, and still contribute knowledge that makes other aquariums safer and easier to understand.

## What works today

### Multi-node local hub

- Polls independent Raspberry Pi tank nodes and combines their camera and device inventories without erasing tank ownership.
- Demonstrates two active tank nodes and keeps their identities, status, feeds, layouts, and controls separate.
- Uses configuration and open payload interfaces so `n=2` can grow without redesigning the hub around a fixed number of tanks.
- Keeps the dashboard, simulator, Sightings, fake nodes, and manual controls useful when physical hardware or the internet is missing.
- Provides one local source of truth for downstream blocks such as SEE SEA TV, spatial modeling, observation, capture, and safe rig control.

### SEE SEA TV

- Rotates available MJPEG feeds every eight seconds.
- Presents one dominant feed with thumbnails, previous/next navigation, and manual pinning.
- Labels every feed with its tank, camera, state, and live status.
- Keeps camera overlays and primary controls clear of the dominant footage.
- Associates the active feed with its simulated camera and viewing direction.

### Local wildlife observation

- Performs low-resolution, low-rate OpenCV motion analysis on assigned camera feeds.
- Reports persistent interesting motion rather than claiming confirmed animal identification.
- Smooths target positions, ignores small contours, and pauses analysis while a rig moves.
- Scores short capture bursts for sharpness and target centering.
- Stores manual and eligible automatic captures in the Sightings album.
- Supports Fish, Shrimp, Snail, Coral, and Unknown labels without requiring AI.

### Inspection cameras and motion control

- **Reels** are positioned directly by the user. Their end-mounted light and narrow form make them the close-inspection option for small, obstructed, or dark spaces.
- **Raydar** surveys a calibrated 12-point circular path, can center persistent motion with bounded movements, and returns to surveying after target loss.
- **Reeflex** exposes bounded motor controls and STOP behavior, but autonomous inspection is still future work.
- Manual controls always preempt Raydar's automated seeking.
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

## Inspection cameras

Reels, Reeflex, and Raydar are equivalent at the system level: each is an inspection-camera instance that can be assigned to a tank, shown in SEE SEA TV, and placed in the simulator. Their difference is how the camera reaches and holds a useful perspective.

| Instance | Placement and purpose | Motion today |
| --- | --- | --- |
| **Reel** | Placed by hand wherever a close view is needed. Its light is attached at the camera end so it can reach into small, dark spaces, look behind structures, and inspect areas a normal exterior camera cannot see. | Manually positioned |
| **Reeflex** | Mounted on an articulated motorized platform for repeatable inspection angles. It is the platform intended to become more autonomous over time. | Direct motor control with safety limits; autonomy is still a goal |
| **Raydar** | Mounted on a pan-and-tilt base to search across a wider area of the tank. | Automated survey and seeking behavior |

## A digital copy of the tanks

The interactive 3D tank model connects camera footage to physical locations and habitat landmarks:

- Two separately labeled tanks can be viewed together or individually.
- `FRONT`, `BACK`, `LEFT`, and `RIGHT` establish orientation.
- Every camera can be represented at its physical location with a field-of-view frustum.
- The active SEE SEA TV source is highlighted so the viewer can connect footage to its real direction.
- Block, slab, rock, pillar, arch, and mound landmarks can be placed on a normalized grid.
- Interior objects can be moved, rotated, scaled, labeled, colored, duplicated, or scattered for quick layout planning.
- Orbit, pan, drag, zoom, overview, and tank-focus controls keep the model navigable on the portrait display.

This creates a practical digital copy of each tank and its surroundings: cameras describe how the habitat is being seen, while structures and landmarks describe what is being seen and where.

### From model to installation

These early 2026 prototypes show the spatial model beside camera feeds and device state, followed by bench and in-water testing. They document an earlier interface generation, rather than a finished view of the current software.

<p align="center">
  <img src="images/readme/see-sea-tv-2026-interface-overview.jpg" alt="Full portrait view of an early 2026 SEE SEA TV interface prototype" width="560">
</p>

*Interface prototyping: bringing the spatial model, camera feeds, device status, and motion focus together on one portrait display.*

<p align="center">
  <img src="images/readme/see-sea-tv-dry-bench-demo.jpg" alt="First 2026 SEE SEA TV dry-bench demo showing the tank simulation above a live camera feed" width="560">
</p>

*A first-generation 2026 dry-bench demo, with the simulated tank and camera geometry above a live feed while the system is tested outside the water.*

<p align="center">
  <img src="images/readme/see-sea-tv-in-water-test-setup.jpg" alt="Two aquarium test setups with cameras and control hardware being tested in water before the display was installed" width="760">
</p>

*The in-water test setup taking shape across both tanks, with camera and motion hardware connected before the portrait display was added.*

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

The [electronics guide](docs/HARDWARE.md) starts with the Pi nodes, PCA9685
controller, MG995 servos and NETGEAR PoE network, including data-path and
power-path diagrams. MG995 motors and the NETGEAR switch are reported used and
tested by the project owner; exact variants, supply ratings and dated results
remain to be recorded. Manufacturer product links are references rather than a
fully verified shopping list. The [design-source register](docs/DESIGN_ORIGINS.md)
tracks missing CAD sources and licenses for the printed assemblies.

<p align="center">
  <img src="images/readme/water-testing-and-hardware-prototypes.jpg" alt="Freshwater test kit beside servo hardware, a controller board, wiring, and an untested 3D-printed dispensing concept" width="760">
</p>

*Freshwater testing and early hardware laid out on the bench. The black 3D-printed dispensing concept shown here was untested and was never installed as a supported system feature.*

### Inside Reeflex

Reeflex uses the [EEZYbotARM Mk2 robotic-arm design published on Autodesk Instructables](https://www.instructables.com/EEZYbotARM-Mk2-3D-Printed-Robot/) as its mechanical basis. Sync Tank applies the arm to underwater camera inspection and automation experiments; the original arm design is not a Sync Tank invention. See the [design-source record and creator credit](docs/DESIGN_ORIGINS.md#reeflex--eezybotarm-mk2) for attribution and outstanding build details.

The platform combines printed mechanical parts, servos, and a PCA9685 controller. The base uses a ring of bearings to support rotation while a geared servo provides motion. Its control board separates multi-channel servo signaling from the Sync controller's higher-level motion-control and safety logic. Increasingly autonomous inspection remains a development goal rather than a current claim. This application does not establish that the original arm, servos, or electronics are submersible; camera mounting, sealing, and aquatic suitability need their own build records.

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

## Architecture

```text
Tank 1 Floaters ──private AP──> Tank 1 Pi ─┐
Tank 2 Floaters ──private AP──> Tank 2 Pi ─┼──PoE Ethernet──> Local Sync hub
Other local cameras and rigs ──────────────┤                       │
Additional tank nodes ─────────────────────┘                       │
                                          ├── SEE SEA TV display
                                          ├── Spatial tank model
                                          ├── Motion observation and alerts
                                          ├── Sightings and field notes
                                          └── Safe inspection-tool control

Chosen clips, Sightings, tests, and fixes ───> Optional community exchange
```

Tank nodes own their private Floater AP, ESP32 JPEG ingest, local camera discovery, servo endpoints, and machine-readable inventory. The Sync hub reaches those nodes through the local PoE Ethernet segment, proxies camera media, maintains combined but separately owned tank layouts, runs optional local analysis, and serves its software blocks. Neither Floaters nor tank Pis require an internet route for normal operation. The current `n=2` deployment validates the pattern while configuration leaves room for more tank nodes. Deployed tank URLs and internal Lighthouse identifiers do not need to change.

Community exchange sits outside the local control path. Nothing in the aquarium should depend on public sharing, and no footage should leave the local hub without an explicit choice.

## Floater network

Floaters are ESP32-S3 Sense still-camera nodes assigned to a specific tank Pi. They do not join the household network, need internet access, or connect directly to the Sync hub. Each Floater joins the private Wi-Fi access point hosted by its owning tank Pi, checks for a command, sends a heartbeat, and uploads a raw JPEG to that Pi on port `8080`.

| Tank | Floater IDs | Private AP | AP address | Tank Pi wired address |
| --- | --- | --- | --- | --- |
| Tank One | `tank-cam-001`, `tank-cam-002` | `TANK_ONE_AP_SSID` | `TANK_ONE_AP_IP/24` | `TANK_ONE_WIRED_IP` |
| Tank Two | `tank-cam-003`, `tank-cam-004` | `TANK_TWO_AP_SSID` | `TANK_TWO_AP_IP/24` | `TANK_TWO_WIRED_IP` |

The deployed AP password is `TANK_AP_PASSWORD`. These AP addresses are stable NetworkManager shared-mode gateway addresses, while individual Floater client addresses may change. The tank Pi stores the latest JPEG and exposes its inventory and image URL to Sync over the isolated wired link.

The tank Pis do not require a normal Wi-Fi or internet connection in deployment. Their `wlan0` interface serves the Floaters; their Ethernet/PoE connection is the only upstream path to the main Sync node at `SYNC_WIRED_IP`. This keeps camera collection working as a fully local chain:

```text
Floater ESP32 ──private tank AP──> tank Pi ──PoE Ethernet──> Sync hub
       JPEG + heartbeat              ingest + ownership       display + storage + analysis
```

In the interface, Floaters remain available as spatial markers. Their still images appear only when a frame changes or a marker is opened, so they do not cover the primary camera view. See [`docs/FLOATER_NETWORK.md`](docs/FLOATER_NETWORK.md) for the endpoint and network handoff.

## Repository map

| Path | Purpose |
| --- | --- |
| [`tank/`](tank/) | Maintained Raspberry Pi tank-node service, ingest receiver, camera registry, controls, setup, and tests |
| [`sync/`](sync/) | Multi-tank controller, SEE SEA TV dashboard, simulator, vision, Sightings, fake nodes, and tests |
| [`docs/`](docs/) | Deployment and tank-specific handoff notes |
| [`archive/`](archive/) | Historical prototypes retained for reference |
| [`images/`](images/) | Project artwork, historical interface images, and README photography |
| [`index.html`](index.html) and [`site/`](site/) | Retro public aquarium website and GitHub Pages setup; live tank services stay local |

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

Simulation is intentionally not treated as proof that a physical rig is safe. Before motorized or automated operation on site:

1. Verify every camera, tank assignment, stream, snapshot, and control URL independently.
2. Calibrate the Raydar center and conservative pan/tilt limits.
3. Exercise STOP before moving through survey waypoints at minimum speed.
4. Verify Reeflex manual poses, limits, STOP, and control preemption under supervision.
5. Compare simulated camera frustums with physical viewing directions.
6. Tune motion thresholds using real reflections, bubbles, plants, and animals.
7. Validate manual and automatic captures without enabling AI analysis.
8. Complete a supervised multi-hour soak before unattended surveying.
9. Make one deliberate Ask the Deep request with a non-sensitive test image and verify the disclosure and stored result.

## Project history

Sync Tank grew through aquarium builds, camera experiments, printed mechanisms, and public exhibits. The 2025 installation brought those pieces together; development around Open Sauce 2026 expanded the work into a shared local controller for two tanks, spatial layouts, and observation tools.

The galleries below preserve that development history, including useful failures. SSTV was the early camera-display component. Its successor, SEE SEA TV, carries that work forward within the wider Sync Tank system.

### Original SSTV — 2025

The original interface concentrated on getting useful pictures onto a screen. It offered a manually selected dominant feed and multi-camera grids identified by their Linux `/dev/video*` sources. Early experiments also drew object-detection boxes and placed generated descriptions or facts directly beneath individual feeds.

<p align="center">
  <img src="images/sstv2.png" alt="Original SSTV single-camera page with manual source-selection controls" width="680">
</p>

*The original dominant-feed view, with direct buttons for switching between local video devices.*

<table>
  <tr>
    <td width="50%"><img src="images/sstv1.png" alt="Original SSTV four-camera grid with an early detected-fish overlay and generated text"></td>
    <td width="50%"><img src="images/sstv3.png" alt="Original SSTV six-camera grid with raw device labels and fish detections"></td>
  </tr>
  <tr>
    <td><em>An early four-camera page combining raw feeds, detection boxes, and generated text.</em></td>
    <td><em>A later six-camera grid showing the range—and inconsistency—of the connected views.</em></td>
  </tr>
</table>

#### Camera and vision experiments

The 2025 archive preserves two development steps: testing alternate SSTV feed layouts and evaluating object-detection output. The detection trials include false positives, not verified animal sightings. These failure cases helped establish why current Sync reports interesting motion without claiming that it has identified an animal.

<table>
  <tr>
    <td width="50%"><img src="images/readme/2025/sstv-fish-dominant-feed.png" alt="Early SSTV dominant feed showing fish near the substrate"></td>
    <td width="50%"><img src="images/readme/2025/sstv-six-camera-caption-experiment.png" alt="Six-camera SSTV experiment with detections and generated captions"></td>
  </tr>
  <tr>
    <td><em>An alternate dominant-feed page focused on fish near the substrate.</em></td>
    <td><em>A six-camera caption experiment combining tank and out-of-tank views.</em></td>
  </tr>
</table>

<table>
  <tr>
    <td width="33%"><img src="images/readme/2025/early-detection-misclassification.png" alt="Apartment camera test incorrectly labeling objects as a jellyfish and penguin"></td>
    <td width="33%"><img src="images/readme/2025/deeplink-diver-detection.png" alt="Early DeepLink object-detection test using an image of a diver in a circular tank"></td>
    <td width="33%"><img src="images/readme/2025/early-two-camera-detection-test.png" alt="Two-camera detection test with incorrect mouse and bed labels"></td>
  </tr>
  <tr>
    <td><em>False-positive jellyfish and penguin labels during a test in the apartment.</em></td>
    <td><em>A DeepLink detection experiment using a diver photograph.</em></td>
    <td><em>A two-camera test recording more useful failure cases.</em></td>
  </tr>
</table>

#### Hardware and live display

SSTV was developed alongside the physical tank. Camera mounts, a local screen, the Reeflex mechanism, and the exhibit monitor were tested as parts of one system rather than as separate demos.

<table>
  <tr>
    <td width="33%"><img src="images/readme/2025/sstv-live-demo-monitor.jpg" alt="SSTV running on a monitor during a live exhibit"></td>
    <td width="33%"><img src="images/readme/2025/tank-side-display.jpg" alt="Small local camera display positioned in front of the aquarium"></td>
    <td width="33%"><img src="images/reeflex.jpg" alt="Full 2025 Reeflex and FPV hardware rig mounted on a tripod"></td>
  </tr>
  <tr>
    <td><em>SSTV running as a live selectable camera display.</em></td>
    <td><em>A small tank-side screen showing the camera perspective beside the real habitat.</em></td>
    <td><em>The full-resolution 2025 Reeflex and FPV hardware assembly.</em></td>
  </tr>
</table>

<p align="center">
  <img src="images/sync.jpg" alt="Original 2025 Sync Tank visual identity with aquarium animals" width="800">
</p>

*The original Sync Tank visual identity used around the 2025 prototype.*

### Building and exhibiting Sync Tank

The public installation brought the aquarium, camera feeds, mechanisms, lighting, power, signage, and control surfaces together. The gallery below follows the exhibit from banner assembly and load-in through the completed booth.

<p align="center">
  <img src="images/synctankimg.jpg" alt="Completed 2025 Sync Tank exhibit with aquarium, mechanisms, local display, and illuminated signage" width="760">
</p>

*The completed 2025 exhibit: instrumented aquarium, local display, moving hardware, power equipment, and the original illuminated backdrop.*

<table>
  <tr>
    <td width="33%"><img src="images/readme/2025/exhibit-banner-assembly.jpg" alt="Original Sync Tank exhibit banner being assembled on the floor"></td>
    <td width="33%"><img src="images/readme/2025/exhibit-load-in-cart.jpg" alt="Aquarium and water containers being moved into the exhibit on a cart"></td>
    <td width="33%"><img src="images/readme/2025/exhibit-neon-signage.png" alt="Completed illuminated aquatic signage around the Sync Tank banner"></td>
  </tr>
  <tr>
    <td><em>Preparing the original banner before installation.</em></td>
    <td><em>Moving the aquarium and water into the venue.</em></td>
    <td><em>The illuminated aquatic backdrop after assembly.</em></td>
  </tr>
</table>

<table>
  <tr>
    <td width="33%"><img src="images/readme/2025/exhibit-table-close.jpg" alt="Close portrait view of the 2025 Sync Tank exhibit table"></td>
    <td width="33%"><img src="images/readme/2025/exhibit-table-wide.jpg" alt="Wide view of the 2025 Sync Tank aquarium and exhibit backdrop"></td>
    <td width="33%"><img src="images/readme/2025/exhibit-team-at-booth.png" alt="Sync Tank team member standing behind the completed 2025 exhibit"></td>
  </tr>
  <tr>
    <td><em>A close view of the working exhibit table.</em></td>
    <td><em>The aquarium and hardware against the completed backdrop.</em></td>
    <td><em>A team member with the finished 2025 installation.</em></td>
  </tr>
</table>

### From SSTV to SEE SEA TV

| Original SSTV | 2026 SEE SEA TV development |
| --- | --- |
| Flat single-feed and camera-grid pages | Portrait multi-tank operations display |
| Raw local video-device names | Tank and camera identities with live state |
| Manual source buttons | Timed rotation, thumbnails, navigation, and pinning |
| Detection boxes and text attached directly to feeds | Local motion events and durable Sightings |
| Camera images without physical context | Tank simulation, camera placement, and field-of-view guides |
| One display host's attached cameras | Multiple tank nodes coordinated by Sync |

The [2026 prototype gallery](#from-model-to-installation) shows the first spatial interfaces. The [current capabilities](#what-works-today) describe the software developed from those experiments.

### Names and additions

| Earlier project language | Current user-visible name | What it means now |
| --- | --- | --- |
| SSTV | **SEE SEA TV** | The rotating, camera-first multi-tank display |
| Lighthouse | **Raydar** | A motorized camera instance with seeking automation; legacy IDs and URLs remain compatible |
| ReefScope / endoscope | **Reel / Reels** | A manually placed, lighted camera instance for inspecting tight spaces |
| REEFLEX | **Reeflex** | A motorized camera instance being developed toward future autonomy |
| Saved frame | **Sighting** | A captured observation with image, source, scores, label, and notes |
| Remote captioning experiments | **Ask the Deep** | A manual-only AI field note for an already captured Sighting |

## License and acknowledgments

Sync Tank is released under the [MIT License](LICENSE). The project is built in the spirit of open science, open hardware, and open curiosity.

Thanks to the robotics, maker, aquarist, and open-source communities whose tools, experiments, and contributions help the project grow.
