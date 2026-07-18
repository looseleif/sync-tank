#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/sync-tank"
VENV_DIR="$APP_DIR/.venv"

mkdir -p "$APP_DIR"
cd "$APP_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

cat > "$APP_DIR/tank-manager.service" <<'EOF'
[Unit]
Description=Sync Tank Manager
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/zero/sync-tank
Environment=SYNC_TANK_STORAGE_DIR=/home/zero/sync-tank/storage
ExecStart=/home/zero/sync-tank/.venv/bin/python /home/zero/sync-tank/tank_manager.py --host 0.0.0.0 --port 8765
Restart=always
RestartSec=5
User=zero

[Install]
WantedBy=multi-user.target
EOF

sudo install -m 0644 "$APP_DIR/tank-manager.service" /etc/systemd/system/tank-manager.service
sudo systemctl daemon-reload
sudo systemctl enable --now tank-manager.service

mkdir -p "$APP_DIR/scripts"
cat > "$APP_DIR/scripts/heartbeat.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
NODE_ID="${NODE_ID:-raspi-sync-tank-001}"
PI_URL="${PI_URL:-http://127.0.0.1:8765}"
while true; do
  curl -sS -X POST "$PI_URL/api/nodes/heartbeat" -H 'Content-Type: application/json' -d "{\"node_id\":\"$NODE_ID\",\"status\":\"online\"}" >/dev/null || true
  sleep 10
done
EOF
chmod +x "$APP_DIR/scripts/heartbeat.sh"
