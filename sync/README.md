# Sync node controller

This directory contains the Sync dashboard and connector processes that coordinate multiple tank nodes.

## Responsibilities

- Render the multi-tank dashboard and camera views.
- Poll tank edge nodes for camera inventory and health.
- Proxy snapshots and streams from tank nodes to the dashboard.
- Organize observations and device state across tanks.
- Receive ESP32 heartbeats and JPEG uploads when deployed as an edge receiver.

## Main entry points

- `tank_manager.py`: dashboard HTTP server, layout state, and camera proxy.
- `edge_receiver.py`: ESP32 heartbeat, command, and JPEG receiver.
- `scripts/start_display_stack.sh`: starts the controller and display stack.
- `scripts/start_wired_display.sh`: configures the tank-node connector polling.
- `static/`: browser dashboard.
- `config/fleet_nodes.json`: known fleet-node configuration.

See `README-pi.md` for deployment and endpoint details.

## Runtime data

Runtime state, uploaded images, generated frames, logs, caches, and virtual environments are intentionally excluded from Git. Each deployed controller creates those locally.
