# Start with one small experiment

## 1. Explore in the browser

Open the [aquarium shop](https://looseleif.github.io/sync-tank/) or
[tank builder](https://looseleif.github.io/sync-tank/builder.html). Neither needs a
Pi, account or connected camera. The builder saves a local draft and exports
JSON. Photos stay in your browser session; they are not uploaded or stored in
the draft. Read [tank modeling](TANK_MODELING.md) for photo alignment and limits.

## 2. Run the local hub without hardware

Clone the repository on your PC or Pi:

```bash
git clone https://github.com/looseleif/sync-tank.git
cd sync-tank
```

Follow [Sync's offline development instructions](../sync/README.md#offline-development)
to run the two fake nodes and register their camera payloads. The browser builder
is a planning sandbox; the local hub owns real camera/node assignments and
hardware control. Do not POST its exported document directly to the hub.

## 3. Identify your electronics

Read the [hardware guide](HARDWARE.md) before choosing parts. It distinguishes
reported hardware from confirmed configurations and links to manufacturer
references. Record exact Pi, PoE receiver, switch and servo variants rather than
buying a similarly named part and assuming it matches.

## 4. Connect one tank

Follow the maintained [tank-node setup](../tank/README.md), with motor power
disconnected during initial service setup. Give the tank and its node distinct,
stable IDs. Verify one camera's snapshot, stream and ownership before adding the
next camera. Use [Floater networking](FLOATER_NETWORK.md) for private Wi-Fi JPEG
ingest and the wired upstream link.

## 5. Model, observe, refine

Start with the tank bounds and one recognizable hide. Add cameras one at a time,
then compare the actual view with the modeled direction. Mark what was measured,
what was observed, and what is still estimated. Add a [design-source record](DESIGN_ORIGINS.md)
as each real part is documented.
