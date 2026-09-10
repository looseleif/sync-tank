# Where the designs came from

[Hardware](HARDWARE.md) | [Tank modeling](TANK_MODELING.md)

This register separates software authorship, purchased hardware, assembly
photographs, and the original source of mechanical designs. A project photo
does not establish authorship or permission to redistribute a printed model.

| Assembly or component | What is available | Origin / outstanding evidence |
| --- | --- | --- |
| Sync hub and tank-node software | Maintained `sync/` and `tank/`, historical `archive/`, repository history | Project implementation under the root MIT license; preserve third-party notices |
| Browser tank builder | `builder.html`, `site/builder.js`; same normalized placement idea and vendored Three.js / OrbitControls as the local display | New photo-reference editor; exports a planning document, not a deployable hub configuration |
| Three.js and OrbitControls | `sync/static/vendor/`, MIT header identifying Three.js authors | [Three.js project and license](https://github.com/mrdoob/three.js); retain its notices in deployment |
| PCA9685 electronics | Driver, defaults, board photograph | Chip family confirmed; breakout brand, PCB revision and schematic source pending; Adafruit link is a reference implementation |
| MG995 servos | Project owner's report of use and testing | Supplier, variant, label photos, power/current measurements and travel calibration pending |
| Raspberry Pi / PoE / NETGEAR network | Deployment docs and owner's report | Pi models, HAT/splitter model, switch model/adapter, port budget and dated test log pending |
| Tonysa underwater camera and analog capture adapter | Owner reports underwater use through a video-to-USB adapter | [Tonysa B08HW881QV](https://www.amazon.com/dp/B08HW881QV); exact adapter/chipset, signal connector, power and immersion evidence pending |
| USB endoscope probes | Owner reports 1 m soft-cord LED borescopes in use | Manufacturer, product link, probe dimensions, video modes and immersion conditions pending |
| Arducam pan/tilt camera | Owner reports metal-case IMX219 autofocus USB model in use | [Matching B029201 reference](https://www.arducam.com/arducam-autofoucs-imx219-usb-camera-b029201.html); installed SKU and measured modes pending |
| REVODATA PoE IP cameras | I704-2-P-HSV6 reported tested; I704-P identified as a candidate only | [Tested model reference](https://www.amazon.com/dp/B0DK737JCZ), [candidate reference](https://www.amazon.com/dp/B0B17FHLV6); firmware, endpoint, codec, integration and test records pending |
| Reeflex robotic arm | Assembly photographs, servo-control software and project owner's identification of the source design | [EEZYbotARM Mk2 on Autodesk Instructables](https://www.instructables.com/EEZYbotARM-Mk2-3D-Printed-Robot/); see the source record below. Exact file revision, local modifications and model-license record pending |
| Raydar pan/tilt mechanics | Control implementation and deployment notes | Mount model, upstream design, local modifications and license not recorded |
| Floater plates and camera housings | Camera-placement model and deployment notes | Mount geometry, fastening/sealing method, material and design source pending |
| Shrimp City structures / hides | Aquarium photographs and simulator landmarks | Individual model sources, licenses, material, print settings and aquatic suitability evidence pending |

No STL, STEP, OpenSCAD, Fusion or Blender design files were found in the tracked
repository during this documentation pass. Do not describe these mechanical
designs as original, licensed for reuse, waterproof, or ready to print until
their individual records support it. The root software license does not relicense
someone else's model.

## Reeflex / EEZYbotARM Mk2

- **Source design:** [EEZYbotARM Mk2 - 3D Printed Robot](https://www.instructables.com/EEZYbotARM-Mk2-3D-Printed-Robot/), published on Autodesk Instructables.
- **Creator credit:** [theGHIZmo](https://www.instructables.com/member/theGHIZmo/), the creator listed on Instructables. Autodesk Instructables hosts the guide; the design is credited to its creator.
- **Use in Sync Tank:** the project owner identifies this design as the basis of the Reeflex arm used for underwater camera inspection and automation experiments. The original mechanics are upstream work; Sync Tank's camera, controller and software integration are project-specific.
- **Still to record:** the exact downloaded model revision and files, their license and required notices, any changed base or arm parts, camera mounts, print materials, sealing methods and test results. Identifying the arm does not establish the origin of every accessory.
- **Scope:** underwater inspection describes this project's application, not a waterproof rating for the upstream design or its electronics. Fully autonomous inspection remains a development goal.

Follow the upstream guide for its model sources and assembly instructions. No
upstream CAD or STL files are redistributed here. Keep the mechanical design's
license separate from this repository's MIT software license; record the terms
for the exact files before adding them or publishing modified models.

## Record each physical design

For each part, add: component ID, source URL, upstream version, designer's
required attribution, license, local changes, editable CAD path, export path,
units, mating hardware, material, print settings and assembly photos. Keep the
project narrative free of personal names; retain attribution where an upstream
license requires it in a dedicated credits record.

## Record a tested build

Use an entry with these fields:

```yaml
build_id: pending
date: pending
node_id: pending
pi_model: pending
controller_board_revision: pending
servo_supplier_and_variant: pending
servo_supply_voltage_and_current_rating: pending
poe_switch_model_and_adapter: pending
poe_receiver_model_and_output: pending
software_commit: pending
test_duration: pending
measured_voltage_under_load: pending
camera_fps_and_age: pending
stop_and_disconnect_result: pending
mechanical_clearance_result: pending
source_links: []
photo_paths: []
failures_and_limitations: []
```

Hardware status is recorded per entry: used, tested by owner report, or candidate.
Remaining identifiers and evidence stay pending until supplied. See the
[camera inventory](HARDWARE.md#camera-inventory-and-feed-paths) and
[tank TODOs](TANK_TODO.md). This keeps the gaps visible without inventing purchase
history or test results.
