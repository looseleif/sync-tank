# Mount and orient the inspection cameras

[Camera inventory](HARDWARE.md#camera-inventory-and-feed-paths) | [Tank TODOs](TANK_TODO.md) | [Photo tank builder](TANK_MODELING.md)

This is a proposed prototype workflow, not a tested mount design or a claim
that the listed probes can remain submerged. Confirm the exact probe's depth,
duration, material and cable-seal limits before wet testing. Keep USB joints,
capture adapters, power supplies, Pi boards and exposed controllers dry.

## Start with a fixed, adjustable support

Prototype one camera on a padded, removable rim clamp carrying an adjustable
rigid support. At its end, a short split cradle holds the camera body, with a
lockable swivel for aim. Keep clamp pressure and loads within the tank maker's
limits; do not drill glass, tighten a screw directly against it, or assume a
rimless tank can take an overhanging load. Select a different independent
support if the tank rim cannot carry it.

The cradle should grip without crushing the probe or loading its cable entry,
and must not cover the lens or LEDs. Use cable strain relief above the water,
a dry-side drip loop, and a routing path clear of animals and moving parts.
The cable should not be the structural support. Record the probe diameter and
minimum cable bend radius before designing the cradle.

```text
  DRY SIDE       USB joint / LED control / capture adapter
                         |
                    drip loop
                         |
  tank rim    [padded clamp + cable strain relief]
                         |  adjustable support
  waterline  ------------|--------------------------------
                         |
                  [lockable swivel]
                         +--[short probe cradle] ---> view
                                    clear lens and LEDs

  Concept only. Wet parts, seals, loads and materials need validation.
```

Begin with a rigid adjustable support rather than a motorized arm: it gives a
repeatable baseline for identifying drift, image quality and model alignment.
Suction cups may be useful for a temporary trial but need a retention/failure
plan before they become the only support. A flexible support also needs a drift
test. Avoid exposed metals, adhesives or printed materials without documented
suitability for the particular aquarium; a printable STL is not that evidence.

## Place one camera from the real view

1. Label the physical probe, cable and capture adapter, then confirm the owning
   `node_id`, `tank_id` and camera ID. Cover or move just that lens to identify
   its live feed. Record USB topology for otherwise identical adapters.
2. Start in a dry bench setup. Verify changing frames, focus and LED controls
   before attempting placement. Only proceed to a supervised, livestock-free
   wet test once the camera and proposed wet materials are suitable.
3. Choose an observation target and a clear cable route. Move the probe body
   with its support; adjust aim at the swivel while watching the actual feed.
   Record the nearest useful focus distance and test with LEDs both off and on
   for glass reflections, backscatter and glare.
4. Use a removable calibration target and at least two known landmarks at
   different depths. Mark which way is up on the probe's dry-side label and
   record image rotation or mirroring. A soft cable can rotate the image even
   when the optical axis still points at the same target.
5. In the planning builder, add an endoscope as **unplaced**, label it with the
   actual camera ID, and set its position from front/side/top observations.
   Adjust yaw/pitch to match the target. Treat the FOV cone as approximate:
   air specifications, including a claimed 170-degree lens, do not establish
   underwater coverage. The builder does not yet expose camera roll or image
   mirroring; note them in the object's observation notes.
6. Lock the physical mount, mark the support position, and take a reference
   image. Use **Done** to leave editing, then export the builder JSON. Reload
   it and compare the pose with the physical view. Builder export is a planning
   file, not a command to the live hub or a robot arm.
7. Check drift after handling the cable, during normal circulation, and after
   removing/reinstalling the cradle. Record time, displacement, angular change,
   leakage/condensation observations and image quality. Stop a test on leakage,
   loose mounting, damaged insulation, excessive heat or unsafe movement.

For optical alignment record the lens center and look-at landmark as well as
the probe-body center used by the builder. A future lens-offset calibration
must bridge those references instead of assuming they are identical.

## When an additional arm is useful

Add an arm only when the fixed-camera trial identifies a specific unreachable
view or necessary inspection path. Compare an extra fixed probe, a manually
adjustable support and a motorized arm for coverage, cable handling, maintenance
and load. Keep actuators dry unless their entire assembly is qualified for the
wet environment; extending EEZYbotARM Mk2 does not make its servos waterproof.

Before physical motion, document workspace limits, glass/animal clearances,
cable snag points, emergency STOP, loss-of-link behavior and recovery after a
restart. Start with an unloaded dry mechanism. Multi-arm collision avoidance,
automatic trajectories and unattended underwater movement are ambitions, not
features demonstrated by the public tank builder.
