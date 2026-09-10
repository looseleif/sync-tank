# Build a tank from observations

[Open the tank builder](https://looseleif.github.io/sync-tank/builder.html)

The public builder is a local, photo-assisted planning tool. It reuses the
project's Three.js renderer library and normalized-coordinate idea, while keeping
its draft separate from live devices. It does not reconstruct hidden surfaces,
infer physical scale, recognize animals or generate a watertight mesh from photos.

## Photo workflow

1. Enter the inside width, height and depth in centimeters. Without a measured
   dimension, scale remains an estimate. An image alone does not establish size.
2. Add a front, right-side or top photo, as square to the glass as practical.
   Avoid wide-angle distortion, reflections and obscured corners where possible.
3. Drag a rectangle around the inside tank area with **Crop tank**. This provides
   an approximate mapping from image pixels to that tank face; it is not camera
   calibration or perspective correction.
4. Add a hide, rock, endoscope or floater. New items start unplaced. Select a
   placement plane and click the 3D tank, or use **Place from photo** and click a
   visible landmark inside the cropped photo.
5. A front photo sets X and Y; a right-side photo sets Z and Y; a top photo sets
   X and Z. The remaining coordinate stays where it was. Use another view to
   refine that depth instead of guessing that the first image measured it.
6. Drag a selected object in the chosen plane or use the continuous position
   sliders. Yellow means selected for editing. **Done** or a double-click in the
   3D viewer deselects it; an ordinary background click does not lose selection.
7. Give cameras their actual IDs as labels, choose a floater mounting face, and
   adjust endoscope yaw/pitch and field of view. The view cone is schematic and
   does not model lens distortion, refraction, occlusion or actual image coverage.
8. Export JSON and keep it with your observations. Draft geometry saves locally
   in the browser when storage is available; reference photos must be reloaded
   after refresh and are never included in exported JSON.

## Coordinate convention

```text
                    Y = 1 (top)
                    |
                    |       Z = 0 (back)
                    |      /
                    |     /
 X = 0 (left) ------+---------------- X = 1 (right)
                   /
                  Z = 1 (front)

 Y = 0 is the floor. Object position is its center.
 Front photo: left -> right is X 0 -> 1; top -> bottom is Y 1 -> 0.
 Right photo: left -> right is Z 1 -> 0; top -> bottom is Y 1 -> 0.
 Top photo: left -> right is X 0 -> 1; top -> bottom is Z 0 -> 1.
```

Physical centimeters are `(x * width, y * height, z * depth)` from the left,
floor and back boundaries. Floater discs lie parallel to their mounting wall
and face straight inward. Interior objects are constrained so their centers and
unrotated bounds remain inside the tank; rotated corners can still intersect
other objects or glass. This is placement planning, not a collision simulation.

## Draft format and transfer

The exported format is `sync-tank-builder`, version 1. It includes tank
dimensions, normalized positions, placed state, item type, label, sizes,
orientation, FOV, mounting face, observation notes, and reference view/crop
metadata. Photos and image bytes are omitted. Imports validate numeric ranges,
known item types and the format/version before replacing a draft.

The local display uses `placement.position`, `placement.target`, `tank_id` and
camera/node identities. A future adapter should explicitly map the builder's
centimeter sizes and normalization into that schema, preserve ownership, and
request review before applying changes. No automatic live-hub import is claimed.

## What a future image reconstruction stage needs

Multiple overlapping views, a known scale reference, camera calibration and
reliable correspondences would be needed for a more automatic pipeline. Glass,
water refraction, reflections, moving animals and hidden surfaces make these
images particularly difficult. Keep inferred geometry separate from measured
geometry and retain confidence and source-image records. The current builder
provides the editable observation model on which that work can build.
