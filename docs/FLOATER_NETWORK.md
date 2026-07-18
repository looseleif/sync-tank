# Floater camera network

Floaters are ESP32-S3 Sense still-camera nodes that report to the Raspberry Pi assigned to their tank. They are not general LAN cameras and do not connect to the main Sync hub or the internet.

## Deployed topology

Each tank Pi has two separate network roles:

- `wlan0` hosts a private 2.4 GHz Wi-Fi access point for that tank's Floaters.
- Ethernet provides the isolated PoE-backed data path from the tank Pi to the main Sync node.

```text
tank-cam-001, tank-cam-002
          │  TANK_ONE_AP_SSID / TANK_ONE_AP_SUBNET/24
          ▼
      Tank One Pi / TANK_ONE_WIRED_IP ─┐
                                    ├── PoE Ethernet ── Sync / SYNC_WIRED_IP
      Tank Two Pi / TANK_TWO_WIRED_IP ─┘
          ▲
          │  TANK_TWO_AP_SSID / TANK_TWO_AP_SUBNET/24
tank-cam-003, tank-cam-004
```

The Floaters have no reason to know the `PRIVATE_IP/24` addresses. Sync has no reason to join either private Wi-Fi AP. The tank Pi is the ownership, ingest, and protocol boundary between the two sides.

## Tank assignments

All network and credential values below are role-based placeholders. Put the real values in untracked local configuration or service environment files; never commit them to the repository.

| Setting | Tank One | Tank Two |
| --- | --- | --- |
| Tank Pi | `tank-pi-001` | `tank-pi-002` |
| Floater IDs | `tank-cam-001`, `tank-cam-002` | `tank-cam-003`, `tank-cam-004` |
| SSID | `TANK_ONE_AP_SSID` | `TANK_TWO_AP_SSID` |
| Password | `TANK_AP_PASSWORD` | `TANK_AP_PASSWORD` |
| AP/gateway address | `TANK_ONE_AP_IP/24` | `TANK_TWO_AP_IP/24` |
| Tank Pi wired address | `TANK_ONE_WIRED_IP/24` | `TANK_TWO_WIRED_IP/24` |
| Sync wired address | `SYNC_WIRED_IP/24` | `SYNC_WIRED_IP/24` |

The AP gateway and wired addresses should be static across boots. Floater client addresses are DHCP leases and should not be used as device identity; the `X-Node-Id` header is the identity.

## Floater-to-tank-Pi requests

Tank One Floaters use `http://TANK_ONE_AP_IP:8080`; Tank Two Floaters use `http://TANK_TWO_AP_IP:8080`.

```text
GET  /api/node/<camera-id>/command
POST /api/node/heartbeat
POST /api/images/upload
```

JPEG upload uses `Content-Type: image/jpeg`, the raw JPEG as the request body, and these identity headers:

```text
X-Node-Id: tank-cam-001
X-Hub-Id: tank-pi-001
```

Tank Two uses its corresponding camera and hub IDs. A Floater may wake, poll its command URL, send a heartbeat, capture or reuse a frame, upload it, and return to its low-power cycle.

## Tank-Pi-to-Sync requests

Sync polls each tank Pi over the wired segment rather than contacting Floaters directly:

```text
Tank One payload: http://TANK_ONE_WIRED_IP:8080/api/pc-hub/payload
Tank Two payload: http://TANK_TWO_WIRED_IP:8080/api/pc-hub/payload
```

The payload advertises Floater ownership, health, and latest-image URLs. Sync can then display, revise, store, or analyze the JPEG while retaining both the tank ID and camera ID.

## Offline behavior

- The tank Pi's AP and ingest receiver continue working without internet service.
- The PoE Ethernet segment carries the local upstream data path and, when used with the required PoE hardware, powers the tank Pi.
- No default gateway or WAN route is required on the tank Pis for normal collection.
- A missing Sync hub does not require a Floater to change endpoints; the owning tank Pi remains its receiver.
- A missing Floater is reported as waiting or stale rather than causing repeated control traffic.

Normal development Wi-Fi may be enabled temporarily for maintenance, but it is not part of the deployed camera path.
