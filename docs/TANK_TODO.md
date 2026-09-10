# Tank development TODOs

[Camera inventory](HARDWARE.md#camera-inventory-and-feed-paths) | [Mounting workflow](CAMERA_MOUNTING.md) | [Tank builder](TANK_MODELING.md)

Open work from the camera inventory, previously reported placement/feed issues,
and future inspection goals. Earlier symptoms are **regression checks to rerun**,
not claims that today's code still has every fault. No task is marked complete
without evidence from the relevant build. No physical tank testing was performed
for this documentation update.

## Review necessary updates first

- [ ] **REV-01 / Review:** walk through this list on each deployed Pi and the
  display hub; record software commit, actual tank/node IDs, available cameras,
  current failures, owner and next review date. Separate configuration fixes,
  missing software support, mechanical work and future research before adding
  more arms or cameras.

## Around each tank

| ID / priority | Location and issue or ambition | TODO / evidence needed to close |
| --- | --- | --- |
| CAM-01 / P0 | Every camera and dry-side connector: incomplete identities | Record model, USB/capture adapter identity or private IP reference, owning node/tank, role and label photo. Verify each physical lens maps to exactly one intended feed; do not equate a marketing model with a runtime camera ID. |
| CAM-02 / P0 | Tonysa capture chain: analog format and adapter unknown | Identify composite versus component, camera power, adapter chipset, input/NTSC settings and Linux modes. Show a changing image through the actual Pi with reconnect and cold-boot checks. |
| CAM-03 / P0 | USB hub: earlier reports of only two of four expected feeds | Reconcile expected connected physical cameras, Linux capture devices, Pi registry and hub display. Check device-name filters, metadata-only V4L2 endpoints, shared USB bandwidth/power and busy capture processes. Test all intended feeds simultaneously; do not create placeholder cameras to reach a count of four. |
| CAM-04 / P0 | Tank ownership: earlier reports of Pi 2 cameras under Pi 1 | Test two nodes with overlapping USB indices. Confirm distinct node-scoped identities, correct tank assignment and no unexpected reassignment after reconnect. Known offline cameras should be clearly marked offline, not counted as live. |
| CAM-05 / P0 | Display: visible image may be stale | Record source timestamps where available, hub receive time, observed FPS, frame age, disconnect indication and recovery. Test moving targets plus unplug/replug; transport arrival alone must not certify a fresh source frame. Agree and record acceptable latency/FPS before calling a stream ready. |
| MNT-01 / P0 | Wet/dry boundary and tank rim: unqualified mounts | Confirm immersion conditions and materials, measure probe/cable dimensions and rim limits, then prototype one removable support/cradle. Pass a supervised livestock-free trial with leakage, cable strain, drift and repeatable positioning records before tank use. |
| MNT-02 / P1 | Interior endoscopes: orientation and usable coverage | Match a labeled live feed to a lens position, look-at landmark, image roll/mirroring, focus distance and observed FOV. Capture reference views after locking the mount; quantify drift after cable handling and circulation. |
| UI-01 / P1 | Placement editor: previously reported jumps and unclear commitment | Repeat face changes followed by X/Y/Z movement; every move must retain the other coordinates. Verify unplaced start, explicit selection, Done/double-click deselection, save feedback, reload persistence and no accidental commands to hardware. Run separately on the planning builder and local live display. |
| UI-02 / P1 | Optical model: body origin is not lens origin | Design lens-offset, roll and image-mirror metadata plus calibration evidence; preserve uncertainty. Do not claim the current schematic cone measures actual underwater coverage. |
| NET-01 / P1 | PoE network: tested IP camera is not yet a verified hub source | For I704-2-P-HSV6, record firmware and local endpoint/codec, verify a decoder/relay into hub stream/snapshot APIs, owner identity, latency, reconnect and Pi CPU load. Keep credentials private. Confirm I704-P is acquired/tested before promoting it from candidate. |
| PTZ-01 / P1 | Dry pan/tilt rig: Arducam aim and cable travel | Confirm SKU and UVC modes, calibrate center and travel limits, test focus at intended distance, cable clearance, STOP and loss-of-link behavior. Compare camera view with recorded rig pose at several positions. |
| ARM-01 / P2 | Perimeter: ambition for additional inspection arms | First map blind spots from fixed probes. Compare extra fixed cameras with passive supports and motorized arms. Record source/license, load, reach, mounting, wet/dry boundary, collision/snag zones and control channel budget before building. |
| AUTO-01 / P2 | Inspection path: automation ambition | Define a supervised task, waypoints, exclusion zones, STOP/preemption and restart behavior. Test dry and unloaded before any approved wet trial; pass a documented safety review before unattended movement. |
| MODEL-01 / P2 | Tank map: track unresolved work spatially | Design optional issue markers tied to tank/camera/object IDs, status, evidence and proposed pose. Distinguish planned from measured positions and persist/export them. The current builder has object notes, not a TODO layer or live-hub transfer adapter. |

P0 means resolve before expanding or relying on the setup; P1 improves reliable
placement and integration; P2 is a future capability. Priorities do not certify
that a physical configuration is safe.

## Per-tank evidence record

Copy this structure into a build record with actual IDs. Keep it as documentation,
not live configuration. Until an issue layer exists, put the task ID in the
relevant builder object's observation notes and link its exported draft here.
Do not invent placed camera coordinates simply to attach a TODO.

```yaml
review_date: pending
reviewer_role: pending
next_review_date: pending
software_commit: pending
tank_id: pending
node_id: pending
issue_id: MNT-01
camera_id: pending
location: pending  # e.g. left rim; proposed until physically verified
status: open
observed_problem: pending
proposed_change: pending
acceptance_check: pending
measured_result: pending
evidence_paths: []
builder_export_path: pending
private_connection_record: pending  # reference only; no passwords or order links
```
